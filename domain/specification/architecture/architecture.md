# Architecture

- GPU: CoreWeave GPU
- Experiment Tracking: W&B
- Evaluation pipeline: W&B 
- LLM observability: W&B



## CoreWeave GPU guidance
- ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-training.coreweave.app
- doc: https://docs.coreweave.com/reference-home
- don't run long training in login-node
- release node after training
- maximum node should be 3
- be careful of the size of assets. be sure to clean data/model after using it (you can leave them during development, but after development, make sure to release them) 
