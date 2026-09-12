# UGC: Universal Graph Coarsening — Reproducibility Fork

A fork of [**katariaMohit/UGC-Universal-Graph-Coarsening**](https://github.com/katariaMohit/UGC-Universal-Graph-Coarsening), the official code for:

> **UGC: Universal Graph Coarsening**
> Mohit Kataria, Sandeep Kumar, Jayadeva — *NeurIPS 2024*
> [Poster / paper](https://nips.cc/virtual/2024/poster/93695) · Indian Institute of Technology Delhi

**Purpose of this fork:** the upstream code assumes a CUDA GPU and pulls heavy
visualisation dependencies at import time, so it fails on CPU-only machines. This
fork makes it **run on CPU** with a modern PyTorch stack, unchanged in method.

All methods and the original implementation are the authors' work. The original README is preserved as [`README_ORIGINAL.md`](README_ORIGINAL.md).

---

## Status

Runs end-to-end on CPU (macOS arm64, Python 3.9.6, torch 2.8.0, PyG 2.6.1).

| Dataset | Coarsening | Model | Test accuracy |
|---------|-----------|-------|---------------|
| cora | 2,708 → 1,392 (48.6% reduction) | `gcn` | **83.24%** |
| cora | 2,708 → 1,392 | `ugc` (APPNP) | **85.08%** |

The LSH coarsening step itself took **~0.02 s**, consistent with the paper's speed claim.

---

## Changes in this fork

All changes are confined to `UGC.py`; the method is untouched.

| Issue | Fix |
|-------|-----|
| `device = 'cuda'` was hardcoded **inside** the training loop, overriding the earlier auto-detection → crashes on machines without CUDA | `device = "cuda" if torch.cuda.is_available() else "cpu"` |
| `from scatter_letters import sl` is a top-level import but is only used by the `--scatter_alphabets` demo mode. It transitively requires `imageio` and OpenCV, so the whole script fails to import without them | Wrapped in `try/except`, falling back to `sl = None` |
| The script saves and reloads whole model objects (`torch.save(model, ...)` / `torch.load(...)`). PyTorch ≥ 2.6 defaults to `weights_only=True` and refuses to unpickle them → `UnpicklingError` | Added a `torch.load(weights_only=False)` compatibility shim |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "numpy<2" torch torch_geometric scipy scikit-learn \
            networkx pygsp matplotlib seaborn pandas
pip install gensim          # required by utils.py
pip install imageio         # optional: only for --scatter_alphabets
pip install scatter_letters # optional: only for --scatter_alphabets
```

Verified with: `torch 2.8.0`, `torch_geometric 2.6.1`, `numpy 1.26.4`, `scipy 1.13.1`, `pygsp 0.6.1`, `gensim 4.4.0`.

> `numpy<2` is pinned because several dependencies still expect the NumPy 1.x ABI.

## Running

```bash
# flagship UGC model (APPNP)
python -W ignore UGC.py --dataset=cora --model_type=ugc \
  --ratio=50 --add_adj_to_node_features=True --alpha=0.19

# vanilla GCN baseline
python -W ignore UGC.py --dataset=cora --model_type=gcn \
  --ratio=50 --add_adj_to_node_features=True --alpha=0.19
```

`--model_type` accepts `gcn`, `sage`, `gat`, `gin`, `ugc` (APPNP), `3wl`.
See [`run.sh`](run.sh) for the full grid of dataset/model/`alpha` combinations from the paper.

Key arguments: `--ratio` (coarsening percentage), `--alpha` (heterophily factor, per-dataset), `--epochs`, `--hidden_units`, `--number_of_projectors` (LSH projectors), `--hash_function` (`dot` / `L1-norm` / `L2-norm`).

## Datasets

**Working — download automatically via PyTorch Geometric:**
`cora`, `citeseer`, `pubmed`, `physics`, `dblp`

**Working — ship with this repository** (`heterophlic_data/`):
`squirrel`, `texas`, `cornell`

**Known issue — `chameleon` and `film`:** these two use hardcoded Windows paths in
`UGC.py`, e.g.

```python
node_feat = r"heterophlic_data\node_feat_cameleon.pt"
file_data_path = r'heterophlic_data\film.mat'
```

Backslash separators do not resolve on macOS or Linux. Changing them to forward
slashes (or `os.path.join`) fixes it; left as-is here to keep this fork's diff
minimal and faithful.

---

## How UGC works (brief)

1. Node features are augmented with the adjacency matrix, weighted by `alpha` (the heterophily factor): `x ← [(1−α)·x , α·A]`.
2. Features are projected against random vectors and **hashed** into bins (an LSH-style scheme); each node's bin, taken as the mode across projectors, is its **supernode**.
3. The coarsened graph is formed by aggregating within supernodes: `A_c = P̂ᵀ A P̂`, with features averaged and labels taken by majority.
4. A standard GNN is trained on the coarsened graph, then evaluated on the **original** graph's test nodes.

Because assignment is by hashing rather than optimisation, coarsening is extremely fast and, unlike spectral methods, extends naturally to heterophilic graphs.

---

## Licence and attribution

The upstream repository carries **no licence file**, so its code remains under the
authors' copyright. This fork exists under the view-and-fork rights granted by
[GitHub's Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service)
for public repositories. No ownership of the original work is claimed.

The three fixes described above are my contribution and may be reused freely (MIT).

If you use this method, please cite the original paper:

```bibtex
@inproceedings{kataria2024ugc,
  title     = {UGC: Universal Graph Coarsening},
  author    = {Kataria, Mohit and Kumar, Sandeep and Jayadeva},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```

*Authors: if you would prefer any part of this handled differently, please open an issue.*
