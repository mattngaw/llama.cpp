# llama-graph-viz

A tool for visualizing the compute graph of llama.cpp models. It loads a model,
reserves a prompt-processing compute graph, and exports it as a
[Graphviz](https://graphviz.org/) DOT file. The default **collapsed** mode
produces a concise, architecture-level diagram; an optional **full** mode dumps
every node and edge unrolled.

## Usage

```bash
# collapsed view (default)
llama-graph-viz -m model.gguf -o graph.dot

# full unrolled view
llama-graph-viz -m model.gguf --full -o graph.dot

# render to image
dot -Tpng graph.dot -o graph.png
dot -Tsvg graph.dot -o graph.svg
```

The tool runs CPU-only — it only needs the graph topology, not actual tensor
computation, so no GPU is required.

### Output

- Default output file: `graph.dot`
- Override with `-o` / `--output`

### Flags

| Flag     | Description                                            |
|----------|--------------------------------------------------------|
| `-m`     | Path to GGUF model file (required)                     |
| `-o`     | Output DOT file path (default: `graph.dot`)            |
| `--full` | Emit the full unrolled graph instead of collapsing layers |

## Architecture

### Modes

**Full mode** (`--full`) delegates directly to ggml's built-in
`ggml_graph_dump_dot`, which emits every node and edge in the raw compute graph.
This is useful for debugging but produces very large graphs for real models
(thousands of nodes).

**Collapsed mode** (default) is the main contribution. It detects repeated
transformer layers, collapses them into a single representative block, and
classifies nodes into Attention and Feed-Forward subclusters. The result is a
compact diagram showing one "Repeating Block (xN layers)" with the full
inter-layer data flow.

### Collapsed mode pipeline

The collapsing algorithm runs in five passes over the compute graph:

#### Pass 1 — Parse tensor names and find representatives

Every node's name is parsed to extract a **canonical name** (layer number
stripped) and a **layer number**. Three naming patterns are recognized:

| Pattern | Example              | Canonical form      |
|---------|----------------------|---------------------|
| `blk.N.rest` | `blk.0.attn_q.weight` | `blk.*.attn_q.weight` |
| `cache_{k,v}_lN...` | `cache_k_l0` | `cache_k` |
| `name-N` / `name-N (suffix)` | `Qcur-0 (RoPE)` | `Qcur (RoPE)` |

For each canonical name, the **lowest-layer instance** is chosen as the
representative tensor. This handles architectures with alternating layer types
(e.g. interleaved SWA/non-SWA attention) by taking the union of all layer
patterns.

The pass also tracks `max_layer` to determine the total layer count.

#### Pass 2 — Partition nodes and build canonical pointer map

Nodes are sorted into three groups based on their position in the graph:

- **Pre-layer nodes**: nodes appearing before any layer-numbered node (e.g.
  embedding lookup, input normalization)
- **Layer nodes**: all nodes with a detected layer number, mapped to their
  lowest-layer representative via a `canon_map`
- **Post-layer nodes**: nodes appearing after the last layer-numbered node (e.g.
  final norm, logits projection)

Only one representative per canonical name is kept in `layer0_nodes` (the
collapsed layer). All other layer instances point to the same representative via
`canon_map`.

#### Pass 3 — Discover and canonicalize leaf tensors

Leaf tensors (model weights, KV cache tensors) are collected by scanning all
node sources for tensors that aren't in the node set. The same
lowest-layer-representative strategy is applied: each leaf gets a canonical
representative via `leaf_canon_map`.

Opaque internal leaves (unnamed `leaf_N` tensors) are identified for filtering.

#### Pass 4 — Collapse trivial ops and deduplicate unnamed nodes

**View collapsing** (Pass 4a): Nodes whose operation is purely structural
(`VIEW`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`, `CONT`) and that have exactly one
canonical source are collapsed out of the graph. Edges are redirected through
them to their source. Redirect chains (A -> B -> C where both are trivial) are
resolved iteratively.

**Unnamed node deduplication** (Pass 4b): Unnamed `node_NNN` nodes remaining in
`post_nodes` (often artifacts of alternating layer patterns) are grouped by a
fingerprint of `(op, type, shape)`. Only one representative per fingerprint
group survives; the rest are collapsed. Surviving representatives are promoted
into `layer0_nodes` since they are part of the repeating pattern.

#### Pass 5 — Collect edges and classify nodes

**Edge collection**: All edges are derived from the raw graph's source pointers,
resolved through both canonical mapping and collapse redirects, then
deduplicated. Back-edges (where the source appears later than the destination in
layer-node ordering) are detected — these represent recurrent connections (e.g.
KV cache feedback) and are rendered distinctly in the output.

**Node classification**: Each layer-0 node is classified as **ATTN**, **FFN**,
or **OTHER** using name-based heuristics:
- ATTN: names matching `attn`, `Kcur`, `Qcur`, `Vcur`, `kqv`, `__fattn__`,
  `cache_k`, `cache_v`, `self_kq`, `rope`
- FFN: names matching `ffn`, `moe`
- OTHER: everything else

A propagation pass then resolves unclassified (OTHER) nodes: if all of a node's
classified neighbors (sources or consumers) belong to a single class, the node
adopts that class. This repeats until stable.

**Bridge node detection**: Nodes that receive input from a different class (e.g.
`ffn_inp` receiving from attention) are marked as bridge nodes and placed outside
subclusters to help Graphviz establish correct vertical ordering between the
Attention and Feed-Forward blocks.

### DOT output structure

The generated DOT file has this structure:

```
digraph G {
  [global settings: TB layout, Helvetica font, compact spacing]

  // Pre-layer computation nodes (light blue)
  // Non-layer leaf tensors (purple)

  subgraph cluster_layer {
    label="Repeating Block (xN layers)"

    // Nodes before attention (grey)

    subgraph cluster_attn {
      label="Attention"
      // Attention computation nodes (white) + leaves (purple)
    }

    // Bridge nodes between attention and FFN (grey)

    subgraph cluster_ffn {
      label="Feed-Forward"
      // FFN computation nodes (white) + leaves (purple)
    }

    // Nodes after FFN (grey)
    // Remaining layer leaves (purple)

    // Invisible ordering edges (attn -> bridge -> ffn)
  }

  // Post-layer computation nodes (light blue)

  // Edges (grey normal, bold blue for back-edges labeled "xN")
}
```

### Visual encoding

| Element | Style | Color |
|---------|-------|-------|
| Computation nodes (pre/post) | Rounded box, filled | Light blue `#E3F2FD` |
| Computation nodes (in layer) | Rounded box, filled | White |
| Computation nodes (other/bridge) | Rounded box, filled | Grey `#E0E0E0` |
| Leaf tensors (weights) | Rounded box, small font | Purple `#E1BEE7` |
| Repeating block cluster | Dashed border | Blue `#1565C0` |
| Attention subcluster | Solid border | Green `#4CAF50` / `#E8F5E9` bg |
| Feed-Forward subcluster | Solid border | Orange `#FF9800` / `#FFF3E0` bg |
| Normal edges | Solid line | Grey `#555555` |
| Back-edges (recurrent) | Bold, with "xN" label | Blue `#1565C0` |

### Node labels

Each node displays:
- **Primary label**: the canonical (layer-stripped) display name, or the op name
  for unnamed `node_NNN` tensors
- **Secondary label** (smaller font): op symbol, data type, and dimensions

### Integration points

The tool integrates with llama.cpp's common infrastructure:

- **`common/common.h`**: adds `LLAMA_EXAMPLE_GRAPH_VIZ` enum value and
  `graph_full` parameter to `common_params`
- **`common/arg.cpp`**: registers `--full` flag and adds `LLAMA_EXAMPLE_GRAPH_VIZ`
  to the `--output` flag's example list
- **`tools/CMakeLists.txt`**: adds the `graph-viz` subdirectory to the build

### Graph construction

The tool builds a prompt-processing compute graph by calling
`llama_graph_reserve` with `n_tokens = min(n_ctx, n_ubatch)` tokens across all
available sequences. This produces the largest representative graph for the model
architecture. The graph is never actually executed — only its topology is
inspected.
