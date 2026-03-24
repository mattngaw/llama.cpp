#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h"
#include "ggml.h"

#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstring>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// parsed tensor name: canonical base (layer stripped) + layer number
struct tensor_id {
    std::string canon; // name with layer number removed
    int         layer; // -1 if no layer detected
};

// extract layer number from tensor name and produce a canonical (layer-stripped) name
static tensor_id parse_tensor_name(const char * name) {
    std::string s(name);

    // pattern B: weight tensors "blk.N.rest"
    {
        std::regex re(R"(^blk\.(\d+)\.(.+)$)");
        std::smatch m;
        if (std::regex_match(s, m, re)) {
            return { "blk.*." + m[2].str(), std::stoi(m[1].str()) };
        }
    }

    // pattern C: cache tensors "cache_k_lN..." or "cache_v_lN..."
    {
        std::regex re(R"(^(cache_[kv])_l(\d+)(.*)$)");
        std::smatch m;
        if (std::regex_match(s, m, re)) {
            return { m[1].str() + m[3].str(), std::stoi(m[2].str()) };
        }
    }

    // pattern A: computation nodes "name-N" or "name-N (suffix)"
    // must anchor the digit group so we don't match partial names
    {
        std::regex re(R"(^(.*)-(\d+)(| .*)$)");
        std::smatch m;
        if (std::regex_match(s, m, re)) {
            return { m[1].str() + m[3].str(), std::stoi(m[2].str()) };
        }
    }

    // fallback: no layer
    return { s, -1 };
}

// produce a display name (layer number stripped) for a tensor
// for unnamed "node_NNN" tensors, use the op name instead
static std::string display_name(const ggml_tensor * t) {
    auto id = parse_tensor_name(t->name);
    // if the canonical name is just "node_NNN" or "node_NNN (suffix)", use op name
    if (std::regex_match(id.canon, std::regex(R"(^node_\d+.*)"))) {
        return ggml_op_name(t->op);
    }
    // if the name is blank or just whitespace + suffix, use op name
    if (id.canon.empty() || id.canon[0] == ' ') {
        return ggml_op_name(t->op);
    }
    return id.canon;
}

// check if a leaf tensor is an opaque internal helper (unnamed leaf_N)
static bool is_opaque_leaf(const ggml_tensor * t) {
    std::string name(t->name);
    if (std::regex_match(name, std::regex(R"(^leaf_\d+$)"))) {
        return true;
    }
    return false;
}

// check if a node has an unnamed/generic "node_NNN" name
static bool is_unnamed_node(const ggml_tensor * t) {
    return std::regex_match(std::string(t->name), std::regex(R"(^node_\d+.*)"));
}

// generate a fingerprint for grouping unnamed nodes: op + type + shape
static std::string node_fingerprint(const ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s_%s_%" PRId64 "_%" PRId64 "_%" PRId64 "_%" PRId64,
             ggml_op_name(t->op), ggml_type_name(t->type),
             t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    return buf;
}

// check if a node is a trivial reshape/view op (structural, not computational)
static bool is_trivial_view_op(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
            return true;
        default:
            return false;
    }
}

// escape a string for use in HTML-like DOT labels
static std::string html_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '<':  out += "&lt;";  break;
            case '>':  out += "&gt;";  break;
            case '&':  out += "&amp;"; break;
            case '"':  out += "&quot;"; break;
            default:   out += c;       break;
        }
    }
    return out;
}

// format tensor dimensions as a compact string
static std::string dims_str(const ggml_tensor * t) {
    char buf[128];
    if (ggml_is_matrix(t)) {
        snprintf(buf, sizeof(buf), "%" PRId64 "x%" PRId64, t->ne[0], t->ne[1]);
    } else {
        snprintf(buf, sizeof(buf), "%" PRId64 "x%" PRId64 "x%" PRId64, t->ne[0], t->ne[1], t->ne[2]);
    }
    return buf;
}

// hash for ggml_tensor pointers used as map keys
struct ptr_hash {
    size_t operator()(const ggml_tensor * p) const { return std::hash<const void *>()(p); }
};

using tensor_set = std::unordered_set<ggml_tensor *, ptr_hash>;
using tensor_map = std::unordered_map<ggml_tensor *, ggml_tensor *, ptr_hash>;
using edge_t     = std::pair<ggml_tensor *, ggml_tensor *>;

// node classification for subclusters
enum class node_class { ATTN, FFN, OTHER };

// classify a node by name heuristics
static node_class classify_by_name(const std::string & name) {
    // attention-related names
    static const std::regex attn_re(
        R"(attn|Kcur|Qcur|Vcur|kqv|__fattn__|cache_k|cache_v|self_kq|rope)",
        std::regex::icase | std::regex::optimize);
    if (std::regex_search(name, attn_re)) {
        return node_class::ATTN;
    }

    // FFN / MoE related names
    static const std::regex ffn_re(
        R"(ffn|moe)",
        std::regex::icase | std::regex::optimize);
    if (std::regex_search(name, ffn_re)) {
        return node_class::FFN;
    }

    return node_class::OTHER;
}

static void graph_dump_dot_collapsed(ggml_cgraph * gf, const char * filename) {
    const int n = ggml_graph_n_nodes(gf);

    // ---- pass 1: parse all node names, find max layer, build canonical lookup ----
    // Use the union of all layers: for each canonical name, pick the lowest-layer
    // representative. This handles architectures with alternating layer types
    // (e.g. interleaved SWA/non-SWA attention).

    tensor_set node_set;
    std::unordered_map<std::string, ggml_tensor *> canon_to_rep; // canon name -> representative tensor
    std::unordered_map<std::string, int> canon_to_min_layer;     // canon name -> lowest layer seen
    int max_layer = -1;

    for (int i = 0; i < n; i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        node_set.insert(t);

        tensor_id id = parse_tensor_name(t->name);
        if (id.layer > max_layer) {
            max_layer = id.layer;
        }
        if (id.layer >= 0) {
            auto it = canon_to_min_layer.find(id.canon);
            if (it == canon_to_min_layer.end() || id.layer < it->second) {
                canon_to_min_layer[id.canon] = id.layer;
                canon_to_rep[id.canon] = t;
            }
        }
    }

    const int layer_count = max_layer + 1;

    // ---- pass 2: partition nodes, build canonical pointer map ----

    tensor_map canon_map;
    std::vector<ggml_tensor *> pre_nodes;
    std::vector<ggml_tensor *> layer0_nodes; // "representative" layer nodes (union of all layer patterns)
    std::vector<ggml_tensor *> post_nodes;

    tensor_set rep_set; // set of representative tensors (to collect them in order)

    bool seen_layer = false;

    for (int i = 0; i < n; i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        tensor_id id = parse_tensor_name(t->name);

        if (id.layer >= 0) {
            seen_layer = true;
            auto it = canon_to_rep.find(id.canon);
            if (it != canon_to_rep.end()) {
                canon_map[t] = it->second;
                // collect representative nodes in graph order
                if (t == it->second && !rep_set.count(t)) {
                    rep_set.insert(t);
                    layer0_nodes.push_back(t);
                }
            } else {
                canon_map[t] = t;
                post_nodes.push_back(t);
            }
        } else {
            canon_map[t] = t;
            if (!seen_layer) {
                pre_nodes.push_back(t);
            } else {
                post_nodes.push_back(t);
            }
        }
    }

    // ---- pass 3: discover leaf tensors, build leaf canonical map ----

    tensor_set all_leaves;
    std::unordered_map<std::string, ggml_tensor *> leaf_canon_lookup;
    tensor_map leaf_canon_map;

    for (int i = 0; i < n; i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = t->src[j];
            if (!src) break;
            if (node_set.count(src)) continue;
            all_leaves.insert(src);
        }
    }

    // pick lowest-layer representative for each leaf canonical name (same approach as nodes)
    std::unordered_map<std::string, int> leaf_canon_min_layer;
    for (ggml_tensor * leaf : all_leaves) {
        tensor_id id = parse_tensor_name(leaf->name);
        auto it = leaf_canon_min_layer.find(id.canon);
        if (it == leaf_canon_min_layer.end() || id.layer < it->second) {
            leaf_canon_min_layer[id.canon] = id.layer;
            leaf_canon_lookup[id.canon] = leaf;
        }
    }
    for (ggml_tensor * leaf : all_leaves) {
        tensor_id id = parse_tensor_name(leaf->name);
        auto it = leaf_canon_lookup.find(id.canon);
        if (it != leaf_canon_lookup.end()) {
            leaf_canon_map[leaf] = it->second;
        } else {
            leaf_canon_map[leaf] = leaf;
        }
    }

    auto get_canon = [&](ggml_tensor * t) -> ggml_tensor * {
        auto it = canon_map.find(t);
        if (it != canon_map.end()) return it->second;
        auto it2 = leaf_canon_map.find(t);
        if (it2 != leaf_canon_map.end()) return it2->second;
        return t;
    };

    // ---- pass 4: identify trivial view nodes to collapse ----
    // A view/reshape/permute/transpose/cont op with exactly one canonical source
    // is purely structural — collapse it out of the graph and redirect edges through it.

    // build canonical adjacency: for each canonical node, its unique canonical sources
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>, ptr_hash> canon_srcs;

    tensor_set layer0_set(layer0_nodes.begin(), layer0_nodes.end());

    for (int i = 0; i < n; i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        ggml_tensor * t_canon = get_canon(t);

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = t->src[j];
            if (!src) break;
            ggml_tensor * src_canon = get_canon(src);
            if (src_canon == t_canon) continue;

            auto & srcs = canon_srcs[t_canon];
            bool already = false;
            for (auto * s : srcs) {
                if (s == src_canon) { already = true; break; }
            }
            if (!already) {
                srcs.push_back(src_canon);
            }
        }
    }

    // identify collapsible view nodes: single source, view/reshape op
    // apply to both layer nodes and post nodes (which may include unnamed view ops)
    tensor_set collapsed_nodes;
    tensor_map collapse_redirect;

    auto try_collapse = [&](ggml_tensor * t) {
        if (!is_trivial_view_op(t)) return;
        auto it = canon_srcs.find(t);
        if (it == canon_srcs.end()) return;
        auto & srcs = it->second;
        if (srcs.size() == 1) {
            collapsed_nodes.insert(t);
            collapse_redirect[t] = srcs[0];
        }
    };

    for (auto * t : layer0_nodes) {
        try_collapse(t);
    }
    for (auto * t : post_nodes) {
        try_collapse(t);
    }
    for (auto * t : pre_nodes) {
        try_collapse(t);
    }

    // resolve redirect chains (A -> B -> C, where both A and B are collapsed)
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto & [node, target] : collapse_redirect) {
            auto it = collapse_redirect.find(target);
            if (it != collapse_redirect.end() && it->second != target) {
                collapse_redirect[node] = it->second;
                changed = true;
            }
        }
    }

    // ---- pass 4b: deduplicate unnamed "node_NNN" nodes by (op, shape) ----
    // Unnamed nodes in post_nodes often come from alternating layer patterns
    // (e.g. SWA vs non-SWA). Group them by fingerprint (op+type+shape) and
    // keep only one representative per group. Surviving representatives are
    // moved into layer0_nodes since they're part of the repeating pattern.

    // track unnamed reps that get promoted from post to layer
    tensor_set promoted_to_layer;

    {
        std::unordered_map<std::string, ggml_tensor *> fingerprint_to_rep;
        for (auto * t : post_nodes) {
            if (collapsed_nodes.count(t)) continue;
            if (!is_unnamed_node(t)) continue;

            std::string fp = node_fingerprint(t);
            auto it = fingerprint_to_rep.find(fp);
            if (it == fingerprint_to_rep.end()) {
                fingerprint_to_rep[fp] = t;
                // move the representative into layer0_nodes
                layer0_nodes.push_back(t);
                promoted_to_layer.insert(t);
            } else {
                // redirect this node to the existing representative
                collapsed_nodes.insert(t);
                collapse_redirect[t] = it->second;
            }
        }
    }

    // re-resolve redirect chains after adding new entries
    changed = true;
    while (changed) {
        changed = false;
        for (auto & [node, target] : collapse_redirect) {
            auto it = collapse_redirect.find(target);
            if (it != collapse_redirect.end() && it->second != target) {
                collapse_redirect[node] = it->second;
                changed = true;
            }
        }
    }

    // helper to resolve a canonical pointer through collapse redirects
    auto resolve = [&](ggml_tensor * t) -> ggml_tensor * {
        auto it = collapse_redirect.find(t);
        return (it != collapse_redirect.end()) ? it->second : t;
    };

    // ---- pass 5: collect and deduplicate edges (with collapse resolution) ----

    std::set<edge_t> edges;
    std::set<edge_t> back_edges;

    // build ordering index for layer0 nodes (non-collapsed only)
    std::unordered_map<ggml_tensor *, int, ptr_hash> layer0_order;
    {
        int idx = 0;
        for (auto * t : layer0_nodes) {
            if (!collapsed_nodes.count(t)) {
                layer0_order[t] = idx++;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        ggml_tensor * t_canon = resolve(get_canon(t));

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = t->src[j];
            if (!src) break;
            ggml_tensor * src_canon = resolve(get_canon(src));
            if (src_canon == t_canon) continue;

            edge_t e = { src_canon, t_canon };
            if (edges.insert(e).second) {
                auto src_it = layer0_order.find(src_canon);
                auto dst_it = layer0_order.find(t_canon);
                if (src_it != layer0_order.end() && dst_it != layer0_order.end()) {
                    if (src_it->second > dst_it->second) {
                        back_edges.insert(e);
                    }
                }
            }
        }
    }

    // ---- invariant checks: verify collapsing preserves all connectivity ----
    {
        int edge_violations = 0;
        int node_violations = 0;
        int raw_edges_checked = 0;

        // 1. Edge preservation: every raw edge must map to a collapsed edge
        //    or have both endpoints in the same equivalence class.
        for (int i = 0; i < n; i++) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = t->src[j];
                if (!src) break;
                raw_edges_checked++;

                ggml_tensor * src_r = resolve(get_canon(src));
                ggml_tensor * dst_r = resolve(get_canon(t));

                if (src_r == dst_r) continue;

                if (!edges.count({ src_r, dst_r })) {
                    LOG_ERR("%s: edge preservation violation: %s -> %s "
                            "(resolved %p -> %p) not in collapsed edge set\n",
                            __func__, src->name, t->name,
                            (const void *)src_r, (const void *)dst_r);
                    edge_violations++;
                }
            }
        }

        // 2. Node coverage: every raw node must map to a canonical representative
        //    that is either in pre_nodes, layer0_nodes, or post_nodes (and not
        //    collapsed away without a valid redirect target).
        tensor_set all_output_nodes;
        for (auto * t : pre_nodes)    { if (!collapsed_nodes.count(t)) all_output_nodes.insert(t); }
        for (auto * t : layer0_nodes) { if (!collapsed_nodes.count(t)) all_output_nodes.insert(t); }
        for (auto * t : post_nodes)   { if (!collapsed_nodes.count(t) && !promoted_to_layer.count(t)) all_output_nodes.insert(t); }
        // promoted nodes are in layer0_nodes, add them too
        for (auto * t : promoted_to_layer) { if (!collapsed_nodes.count(t)) all_output_nodes.insert(t); }

        for (int i = 0; i < n; i++) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            ggml_tensor * rep = resolve(get_canon(t));
            if (!all_output_nodes.count(rep)) {
                LOG_ERR("%s: node coverage violation: %s (resolved to %p) "
                        "has no representative in output node sets\n",
                        __func__, t->name, (const void *)rep);
                node_violations++;
            }
        }

        if (edge_violations > 0 || node_violations > 0) {
            LOG_ERR("%s: invariant check FAILED: %d edge violations, %d node violations\n",
                    __func__, edge_violations, node_violations);
        } else {
            LOG_INF("%s: invariant check passed (%d raw edges verified, %d raw nodes verified)\n",
                    __func__, raw_edges_checked, n);
        }
    }

    // ---- collect canonical leaves ----

    tensor_set emitted_leaves;
    tensor_set layer0_leaves;
    tensor_set nonlayer_leaves;
    tensor_set opaque_leaves;

    for (auto & [src, dst] : edges) {
        for (ggml_tensor * ep : { src, dst }) {
            if (!node_set.count(ep) && !emitted_leaves.count(ep)) {
                emitted_leaves.insert(ep);
                if (is_opaque_leaf(ep)) {
                    opaque_leaves.insert(ep);
                    continue;
                }
                tensor_id id = parse_tensor_name(ep->name);
                if (id.layer == 0) {
                    layer0_leaves.insert(ep);
                } else {
                    nonlayer_leaves.insert(ep);
                }
            }
        }
    }

    // ---- classify layer0 nodes into attn / ffn / other ----
    // First pass: classify by name. Second pass: propagate to unclassified nodes via connectivity.

    std::unordered_map<ggml_tensor *, node_class, ptr_hash> node_classes;

    // classify non-collapsed layer0 nodes by name
    for (auto * t : layer0_nodes) {
        if (collapsed_nodes.count(t)) continue;
        node_classes[t] = classify_by_name(display_name(t));
    }

    // build consumer adjacency (reverse of canon_srcs) for forward propagation
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>, ptr_hash> canon_consumers;
    for (auto * t : layer0_nodes) {
        if (collapsed_nodes.count(t)) continue;
        auto it = canon_srcs.find(t);
        if (it == canon_srcs.end()) continue;
        for (auto * src : it->second) {
            src = resolve(src);
            if (node_classes.count(src)) {
                canon_consumers[src].push_back(t);
            }
        }
    }

    // propagate: if an OTHER node has all its classified neighbors (sources OR consumers)
    // in one class, adopt that class. Repeat until stable.
    changed = true;
    while (changed) {
        changed = false;
        for (auto * t : layer0_nodes) {
            if (collapsed_nodes.count(t)) continue;
            if (node_classes[t] != node_class::OTHER) continue;

            node_class inferred = node_class::OTHER;
            bool has_classified = false;

            // check sources
            auto src_it = canon_srcs.find(t);
            if (src_it != canon_srcs.end()) {
                for (auto * src : src_it->second) {
                    src = resolve(src);
                    auto it = node_classes.find(src);
                    if (it == node_classes.end() || it->second == node_class::OTHER) continue;
                    if (!has_classified) {
                        inferred = it->second;
                        has_classified = true;
                    } else if (inferred != it->second) {
                        inferred = node_class::OTHER;
                        break;
                    }
                }
            }

            // if sources weren't conclusive, also check consumers
            if (!has_classified || inferred == node_class::OTHER) {
                has_classified = false;
                inferred = node_class::OTHER;
                auto cons_it = canon_consumers.find(t);
                if (cons_it != canon_consumers.end()) {
                    for (auto * con : cons_it->second) {
                        auto it = node_classes.find(con);
                        if (it == node_classes.end() || it->second == node_class::OTHER) continue;
                        if (!has_classified) {
                            inferred = it->second;
                            has_classified = true;
                        } else if (inferred != it->second) {
                            inferred = node_class::OTHER;
                            break;
                        }
                    }
                }
            }

            if (has_classified && inferred != node_class::OTHER) {
                node_classes[t] = inferred;
                changed = true;
            }
        }
    }

    // detect "bridge" nodes: classified nodes whose sources include a different class
    // (e.g. ffn_inp receives from attention). These should be outside subclusters
    // to help dot establish the vertical ordering between subclusters.
    tensor_set bridge_nodes;
    for (auto * t : layer0_nodes) {
        if (collapsed_nodes.count(t)) continue;
        node_class my_class = node_classes[t];
        if (my_class == node_class::OTHER) continue;

        auto src_it = canon_srcs.find(t);
        if (src_it == canon_srcs.end()) continue;
        for (auto * src : src_it->second) {
            src = resolve(src);
            auto it = node_classes.find(src);
            if (it != node_classes.end() && it->second != node_class::OTHER && it->second != my_class) {
                bridge_nodes.insert(t);
                break;
            }
        }
    }

    // partition layer0 nodes and leaves by class
    std::vector<ggml_tensor *> layer0_attn_nodes, layer0_ffn_nodes, layer0_other_nodes;
    std::vector<ggml_tensor *> layer0_attn_leaves, layer0_ffn_leaves, layer0_other_leaves;

    for (auto * t : layer0_nodes) {
        if (collapsed_nodes.count(t)) continue;
        // bridge nodes go outside subclusters
        if (bridge_nodes.count(t)) {
            layer0_other_nodes.push_back(t);
            continue;
        }
        switch (node_classes[t]) {
            case node_class::ATTN: layer0_attn_nodes.push_back(t); break;
            case node_class::FFN:  layer0_ffn_nodes.push_back(t);  break;
            default:               layer0_other_nodes.push_back(t); break;
        }
    }

    // classify leaves by which nodes they connect to
    for (auto * t : layer0_leaves) {
        std::string dname = display_name(t);
        node_class cls = classify_by_name(dname);
        switch (cls) {
            case node_class::ATTN: layer0_attn_leaves.push_back(t); break;
            case node_class::FFN:  layer0_ffn_leaves.push_back(t);  break;
            default:               layer0_other_leaves.push_back(t); break;
        }
    }

    // ---- emit DOT ----

    FILE * fp = fopen(filename, "w");
    if (!fp) {
        LOG_ERR("failed to open %s for writing\n", filename);
        return;
    }

    auto emit_comp_node = [&](FILE * f, const ggml_tensor * t, const char * indent,
                              const char * fill_color) {
        std::string dname = html_escape(display_name(t));
        std::string dims  = dims_str(t);
        std::string opsym = html_escape(ggml_op_symbol(t->op));
        fprintf(f, "%s\"%p\" [\n", indent, (const void *)t);
        fprintf(f, "%s  shape=box; style=\"filled,rounded\"; fillcolor=\"%s\";\n",
                indent, fill_color);
        fprintf(f, "%s  label=<%s<BR/><FONT POINT-SIZE=\"9\">%s %s [%s]</FONT>>;\n",
                indent, dname.c_str(), opsym.c_str(), ggml_type_name(t->type), dims.c_str());
        fprintf(f, "%s];\n", indent);
    };

    auto emit_leaf_node = [&](FILE * f, const ggml_tensor * t, const char * indent) {
        std::string dname = html_escape(display_name(t));
        std::string dims  = dims_str(t);
        fprintf(f, "%s\"%p\" [\n", indent, (const void *)t);
        fprintf(f, "%s  shape=box; style=\"filled,rounded\"; fillcolor=\"#E1BEE7\"; fontsize=9;\n", indent);
        fprintf(f, "%s  label=<%s<BR/><FONT POINT-SIZE=\"8\">%s [%s]</FONT>>;\n",
                indent, dname.c_str(), ggml_type_name(t->type), dims.c_str());
        fprintf(f, "%s];\n", indent);
    };

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  rankdir=TB;\n");
    fprintf(fp, "  compound=true;\n");
    fprintf(fp, "  ranksep=0.5;\n");
    fprintf(fp, "  nodesep=0.25;\n");
    fprintf(fp, "  fontname=\"Helvetica\";\n");
    fprintf(fp, "  node [fontname=\"Helvetica\"; fontsize=11];\n");
    fprintf(fp, "  edge [color=\"#555555\"];\n");
    fprintf(fp, "\n");

    // pre-layer nodes
    for (auto * t : pre_nodes) {
        if (collapsed_nodes.count(t)) continue;
        emit_comp_node(fp, t, "  ", "#E3F2FD");
    }

    // non-layer leaves
    for (auto * t : nonlayer_leaves) {
        emit_leaf_node(fp, t, "  ");
    }

    if (!pre_nodes.empty() || !nonlayer_leaves.empty()) {
        fprintf(fp, "\n");
    }

    // ---- layer cluster with attention and FFN subclusters ----
    fprintf(fp, "  subgraph cluster_layer {\n");
    fprintf(fp, "    label=<<B>Repeating Block</B> (x%d layers)>;\n", layer_count);
    fprintf(fp, "    style=dashed; color=\"#1565C0\"; fontcolor=\"#1565C0\"; fontsize=13;\n");
    fprintf(fp, "    bgcolor=\"#FAFAFA\";\n");
    fprintf(fp, "\n");

    // Build position index for layer0_nodes to partition "other" nodes
    // into before-attn, between attn/ffn, and after-ffn sections
    std::unordered_map<ggml_tensor *, int, ptr_hash> node_order;
    {
        int idx = 0;
        for (auto * t : layer0_nodes) {
            if (!collapsed_nodes.count(t)) {
                node_order[t] = idx++;
            }
        }
    }

    int first_attn_pos = INT_MAX, last_attn_pos = -1;
    int first_ffn_pos = INT_MAX, last_ffn_pos = -1;
    for (auto * t : layer0_attn_nodes) {
        int p = node_order[t];
        if (p < first_attn_pos) first_attn_pos = p;
        if (p > last_attn_pos) last_attn_pos = p;
    }
    for (auto * t : layer0_ffn_nodes) {
        int p = node_order[t];
        if (p < first_ffn_pos) first_ffn_pos = p;
        if (p > last_ffn_pos) last_ffn_pos = p;
    }

    std::vector<ggml_tensor *> other_before, other_between, other_after;
    for (auto * t : layer0_other_nodes) {
        int p = node_order[t];
        if (p < first_attn_pos) {
            other_before.push_back(t);
        } else if (p > last_attn_pos && p < first_ffn_pos) {
            other_between.push_back(t);
        } else {
            other_after.push_back(t);
        }
    }

    // emit: before-attn nodes, attn cluster, between nodes, ffn cluster, after nodes
    for (auto * t : other_before) {
        emit_comp_node(fp, t, "    ", "#E0E0E0");
    }

    // attention subcluster
    if (!layer0_attn_nodes.empty()) {
        fprintf(fp, "    subgraph cluster_attn {\n");
        fprintf(fp, "      label=<<I>Attention</I>>;\n");
        fprintf(fp, "      style=solid; color=\"#4CAF50\"; fontcolor=\"#2E7D32\"; fontsize=11;\n");
        fprintf(fp, "      bgcolor=\"#E8F5E9\";\n");
        fprintf(fp, "\n");
        for (auto * t : layer0_attn_nodes) {
            emit_comp_node(fp, t, "      ", "white");
        }
        for (auto * t : layer0_attn_leaves) {
            emit_leaf_node(fp, t, "      ");
        }
        fprintf(fp, "    }\n\n");
    }

    // bridge nodes between attention and FFN
    for (auto * t : other_between) {
        emit_comp_node(fp, t, "    ", "#E0E0E0");
    }

    // FFN subcluster
    if (!layer0_ffn_nodes.empty()) {
        fprintf(fp, "    subgraph cluster_ffn {\n");
        fprintf(fp, "      label=<<I>Feed-Forward</I>>;\n");
        fprintf(fp, "      style=solid; color=\"#FF9800\"; fontcolor=\"#E65100\"; fontsize=11;\n");
        fprintf(fp, "      bgcolor=\"#FFF3E0\";\n");
        fprintf(fp, "\n");
        for (auto * t : layer0_ffn_nodes) {
            emit_comp_node(fp, t, "      ", "white");
        }
        for (auto * t : layer0_ffn_leaves) {
            emit_leaf_node(fp, t, "      ");
        }
        fprintf(fp, "    }\n\n");
    }

    // nodes after FFN
    for (auto * t : other_after) {
        emit_comp_node(fp, t, "    ", "#E0E0E0");
    }

    // remaining layer leaves
    for (auto * t : layer0_other_leaves) {
        emit_leaf_node(fp, t, "    ");
    }

    // invisible ordering edges to enforce: attn above FFN, bridge nodes between
    if (!layer0_attn_nodes.empty() && !other_between.empty()) {
        fprintf(fp, "    \"%p\" -> \"%p\" [style=invis; weight=10];\n",
                (const void *)layer0_attn_nodes.back(),
                (const void *)other_between.front());
    }
    if (!other_between.empty() && !layer0_ffn_nodes.empty()) {
        fprintf(fp, "    \"%p\" -> \"%p\" [style=invis; weight=10];\n",
                (const void *)other_between.back(),
                (const void *)layer0_ffn_nodes.front());
    }
    if (other_between.empty() && !layer0_attn_nodes.empty() && !layer0_ffn_nodes.empty()) {
        fprintf(fp, "    \"%p\" -> \"%p\" [style=invis; weight=10];\n",
                (const void *)layer0_attn_nodes.back(),
                (const void *)layer0_ffn_nodes.front());
    }

    fprintf(fp, "  }\n\n");

    // post-layer nodes
    for (auto * t : post_nodes) {
        if (collapsed_nodes.count(t)) continue;
        if (promoted_to_layer.count(t)) continue;
        emit_comp_node(fp, t, "  ", "#E3F2FD");
    }

    if (!post_nodes.empty()) {
        fprintf(fp, "\n");
    }

    // ---- edges ----
    for (auto & [src, dst] : edges) {
        if (opaque_leaves.count(src) || opaque_leaves.count(dst)) continue;
        // skip edges to/from collapsed nodes (shouldn't exist after resolution, but be safe)
        if (collapsed_nodes.count(src) || collapsed_nodes.count(dst)) continue;

        if (back_edges.count({ src, dst })) {
            fprintf(fp, "  \"%p\" -> \"%p\" [\n"
                        "    style=bold; color=\"#1565C0\"; constraint=false;\n"
                        "    label=<<B>x%d</B>>; fontcolor=\"#1565C0\"; fontsize=10;\n"
                        "  ];\n",
                    (const void *)src, (const void *)dst, layer_count);
        } else {
            fprintf(fp, "  \"%p\" -> \"%p\";\n",
                    (const void *)src, (const void *)dst);
        }
    }

    fprintf(fp, "}\n");
    fclose(fp);

    int post_visible = 0;
    for (auto * t : post_nodes) {
        if (!collapsed_nodes.count(t) && !promoted_to_layer.count(t)) post_visible++;
    }
    int pre_visible = 0;
    for (auto * t : pre_nodes) {
        if (!collapsed_nodes.count(t)) pre_visible++;
    }
    int visible_nodes = pre_visible + (int)(layer0_attn_nodes.size() +
                              layer0_ffn_nodes.size() + layer0_other_nodes.size()) +
                              post_visible;
    int visible_leaves = (int)(emitted_leaves.size() - opaque_leaves.size());

    LOG_INF("%s: collapsed %d layers, %d nodes + %d leaves -> %d nodes + %d leaves "
            "(%d view-collapsed, %d opaque filtered)\n",
            __func__, layer_count, n, (int)all_leaves.size(),
            visible_nodes, visible_leaves,
            (int)collapsed_nodes.size(), (int)opaque_leaves.size());
}

int main(int argc, char ** argv) {
    common_params params;
    params.out_file  = "graph.dot";
    params.warmup    = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_GRAPH_VIZ)) {
        return 1;
    }

    common_init();

    // load CPU-only since we just need the graph topology, not actual computation
    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    params.devices      = { cpu_device, nullptr };
    params.fit_params   = false;
    params.n_gpu_layers = 0;

    auto init_result = common_init_from_params(params);
    if (!init_result) {
        LOG_ERR("failed to initialize model/context\n");
        return 1;
    }

    llama_context * ctx = init_result->context();

    const uint32_t n_seqs   = llama_n_seq_max(ctx);
    const uint32_t n_tokens = std::min(llama_n_ctx(ctx), llama_n_ubatch(ctx));

    // build prompt-processing graph (many tokens)
    auto * gf = llama_graph_reserve(ctx, n_tokens, n_seqs, n_tokens);
    if (!gf) {
        LOG_ERR("failed to reserve graph\n");
        return 1;
    }

    LOG_INF("graph: %d nodes\n", ggml_graph_n_nodes(gf));

    if (params.graph_full) {
        ggml_graph_dump_dot(gf, nullptr, params.out_file.c_str());
    } else {
        graph_dump_dot_collapsed(gf, params.out_file.c_str());
    }

    LOG_INF("graph written to %s\n", params.out_file.c_str());

    return 0;
}
