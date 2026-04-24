# Artist Popularity Prediction using GNN

This project uses Graph Neural Networks (GraphSAGE) to predict artist popularity based on:

- Followers
- Collaborations
- Genre

## Features
- Graph-based modeling
- Node classification
- New artist simulation

## Results
- ~96% accuracy
- Clear clustering in embedding space

## Testing
- Can change what genre in the load_and_filter() and then in the main() you can edit the parameters to run simulation
## How to Run
```bash
pip install -r requirements.txt
python artist_gnn.py
# Torch-geometric may need special install steps depending on system
pip install torch
pip install torch-geometrics
