# UGC-Universal-Graph-Coarsening
This is the official repository of [UGC](https://nips.cc/virtual/2024/poster/93695): Universal Graph Coarsening accepted in 38 Conference on Neural Information Processing Systems ( [NeurIPS24](https://neurips.cc/Conferences/2024/CallForPapers)), held at the Vancouver Convention Center.

UGC is a graph coarsening framework. Inspired by Locality Sensitive Hashing (LSH), UGC uses hashing function to allocate node of original graph to supernode of coarsened graph, achieving both speed and property preservation of the original graph. UGC is 4x to 15x faster, has lower eigen-error, and yields superior performance on downstream processing tasks even at 70% coarsening ratios.

## Key Features
- **High Efficiency:** Leveraging LSH-inspired hashing, UGC is the fastest methods for graph coarsening.
- **Heterophilic Dataset Support:** It’s the first coarsening method for heterophilic graphs, addressing their unique structural challenges.
- **Improved Node Classification Accuracy:** Post-coarsening, UGC enhances node classification accuracy, making it valuable for real-world applications.


## Dependencies
Install required dependencies from requirement.txt file.

## Run
- UGC is model agnostic and currently supports Vanilla GCN, GraphSage, GIN, GAT, APPNP GCN and 3WL gnn models.
- Experiments includes both homophilic and heterophilic datasets.

To run UGC use run.sh file.

- python UGC.py --dataset=cora --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.19
- python UGC.py --dataset=dblp --model_type=ugc --ratio=50 --add_adj_to_node_features=True --alpha=0.18
- python UGC.py --dataset=pubmed --model_type=3wl --ratio=50 --add_adj_to_node_features=True --alpha=0.20
- python UGC.py --dataset=physics --model_type=sage --ratio=50 --add_adj_to_node_features=True --alpha=0.07
- python UGC.py --dataset=squirrel --model_type=gat --ratio=50 --add_adj_to_node_features=True --alpha=0.78
- python UGC.py --dataset=chameleon --model_type=gin --ratio=50 --add_adj_to_node_features=True --alpha=0.75
- python UGC.py --dataset=texas --model_type=ugc --ratio=50 --add_adj_to_node_features=True --alpha=0.91
- python UGC.py --dataset=film --model_type=ugc --ratio=50 --add_adj_to_node_features=True --alpha=0.78
- python UGC.py --dataset=cornell --model_type=ugc --ratio=50 --add_adj_to_node_features=True --alpha=0.70
