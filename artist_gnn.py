import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
import ast
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_and_filter_data():
    edges_df = pd.read_csv("edges.csv")
    nodes_df = pd.read_csv("nodes.csv")

    # Filter to keep the graph smaller and more meaningful, can change the genre to something else if desired
    filtered_nodes = nodes_df[
        (nodes_df["genres"].str.contains("hip hop", case=False, na=False)) &
        (nodes_df["popularity"] > 40)
    ].copy()

    print("Filtered artists:", len(filtered_nodes))

    return edges_df, filtered_nodes

def count_chart_hits(val):
    if pd.isna(val) or val == "":
        return 0
    try:
        parsed = ast.literal_eval(val)
        return len(parsed)
    except:
        return 0


def prepare_node_features(nodes_df):
    nodes_df = nodes_df.copy()

    nodes_df["followers"] = nodes_df["followers"].fillna(0)
    nodes_df["chart_hits"] = nodes_df["chart_hits"].fillna("")
    nodes_df["genres"] = nodes_df["genres"].fillna("")

    nodes_df["chart_hits_count"] = nodes_df["chart_hits"].apply(count_chart_hits)

    def split_genres(genre_str):
        # Needs to be a list so that MLB can process it. 
        if pd.isna(genre_str) or genre_str == "":
            return []

        try:
            parsed = ast.literal_eval(genre_str)

            if isinstance(parsed, list):
                cleaned_genres = []
                for g in parsed:
                    g = str(g).strip()
                    if g != "":
                        cleaned_genres.append(g)
                return cleaned_genres

        except:
            pass
        # Backup parser because some of the genre strings are not well-formed lists. 
        # This will try to split by comma and clean up extra characters.
        cleaned_genres = []
        for g in str(genre_str).split(","):
            g = g.strip().strip("'").strip('"').strip("[]")
            if g != "":
                cleaned_genres.append(g)

        return cleaned_genres
    
    genre_lists = nodes_df["genres"].apply(split_genres)

    # Debugging: print some of the raw genre strings and the parsed lists to verify correctness
    # print("\nFirst 10 raw genre values:")

    # print(nodes_df["genres"].head(10).tolist())

    # print("\nFirst 10 parsed genre lists:")

    # print(genre_lists.head(10).tolist())

    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(genre_lists)

    # Debugging: print some info about the genres learned by mlb
    # print("\nFirst 100 genres learned by mlb:")

    # print(mlb.classes_[:100])

    # print("\nTotal number of genres learned:", len(mlb.classes_))

    # print("\nDoes mlb know 'hip hop'?", "hip hop" in mlb.classes_)

    # print("Does mlb know 'trap'?", "trap" in mlb.classes_)

    numeric_features = nodes_df[["followers", "chart_hits_count"]].values.astype(float)

    # Standardize the numeric features to have mean=0 and std=1, which can help the GNN learn better
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(numeric_features)

    # Combine numeric and genre features into a single feature matrix
    x = np.hstack([numeric_features, genre_features]).astype(np.float32)
    
    # Create binary labels: 1 if popularity >= 70, else 0
    y = (nodes_df["popularity"] >= 70).astype(int).values

    return x, y, mlb, scaler


def build_graph(edges_df, nodes_df):
    # Keep only artists that survived filtering
    valid_ids = set(nodes_df["spotify_id"])

    edges_filtered = edges_df[
        edges_df["id_0"].isin(valid_ids) & edges_df["id_1"].isin(valid_ids)
    ].copy()

    # Remove self-loops
    edges_filtered = edges_filtered[edges_filtered["id_0"] != edges_filtered["id_1"]]

    print("Filtered edges:", len(edges_filtered))

    # Map spotify_id -> integer node index
    id_to_idx = {artist_id: i for i, artist_id in enumerate(nodes_df["spotify_id"])}

    #My code for mapping for easier explanantion. For dictionary mapping. 
    # id_to_id = {}
    # for i in artist_id in enumerate(nodes_df["spotify_id"]):
    #     id_to_id[artist_id] = i

    # Convert edge list to integer index pairs
    edge_pairs = []
    for _, row in edges_filtered.iterrows():
        artist_1 = id_to_idx[row["id_0"]]
        artist_2 = id_to_idx[row["id_1"]]
        edge_pairs.append((artist_1, artist_2))
        edge_pairs.append((artist_2, artist_1))  # undirected graph for PyG
        # Making sure that I show the collaboartions in both directions for the model. 

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    # This converts python list to list of edges in the Pytorch Geometric format.
    # The shape before transpose is (num_edges, 2) where each row is (source, target). 
    # After transpose, it becomes (2, num_edges) which is the format expected by PyG.
    # First row = source nodes, second row = target nodes.

    return edge_index, id_to_idx


# Creating a PyTorch Geometric Data object from the features, labels, and edge index. This will be used for training the GNN.
def create_data_object(x, y, edge_index):
    x_tensor = torch.tensor(x, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)

    num_nodes = x_tensor.shape[0]

    # Train/val/test masks
    indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=y, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=42
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        logits = self.classifier(x)
        return logits, x  # x here is the embedding


def train_model(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        out_channels=2
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0
    best_state = None

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        logits, embeddings = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, embeddings = model(data.x, data.edge_index)
            preds = logits.argmax(dim=1)

            train_acc = (preds[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            val_acc = (preds[data.val_mask] == data.y[data.val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, device


def evaluate_model(model, data, device):
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        logits, embeddings = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = preds[data.test_mask].cpu().numpy()

    print("\nTest classification report:")
    print(classification_report(y_true, y_pred))

    return embeddings.cpu().numpy(), preds.cpu().numpy()


def find_similar_artists(embeddings, nodes_df, artist_name, top_k=5):
    if artist_name not in set(nodes_df["name"]):
        print(f"Artist '{artist_name}' not found.")
        return

    target_idx = nodes_df.index[nodes_df["name"] == artist_name][0]

    nbrs = NearestNeighbors(n_neighbors=top_k + 1, metric="cosine")
    nbrs.fit(embeddings)

    distances, indices = nbrs.kneighbors([embeddings[target_idx]])

    print(f"\nArtists most similar to {artist_name}:")
    for idx in indices[0][1:]:
        print(nodes_df.iloc[idx]["name"])


    
#Encode a new arist and find similar artists
def encode_new_artist(new_artist, scaler, mlb):
    followers = float(new_artist.get("followers", 0))

    if "chart_hits_count" in new_artist:
        chart_hits = float(new_artist["chart_hits_count"])
    else:
        chart_hits = float(count_chart_hits(new_artist.get("chart_hits", "")))

    genres = new_artist.get("genres", [])
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(",") if g.strip()]

    numeric = np.array([[followers, chart_hits]], dtype=float)
    numeric_scaled = scaler.transform(numeric)

    genre_encoded = mlb.transform([genres])

    features_vec = np.hstack([numeric_scaled, genre_encoded]).astype(np.float32)
    return features_vec
    

def predict_new_artist_popularity(
    model,
    data,
    filtered_nodes,
    scaler,
    mlb,
    new_artist,
    collaborator_names,
    device,
    top_k=5
):
    """
    Insert a hypothetical new artist into the graph and predict:
      - popularity class
      - probability of high popularity
      - nearest artists in embedding space

    Parameters
    ----------
    model : trained GraphSAGE model
    data : original PyG Data object
    filtered_nodes : DataFrame used to build the graph
    scaler : fitted StandardScaler
    mlb : fitted MultiLabelBinarizer
    new_artist : dict with artist metadata
    collaborator_names : list of existing artist names to connect to
    device : torch device
    top_k : number of similar artists to return
    """
    model.eval()

    # ---- Step 1: encode the new artist into the same feature space ----
    new_x_np = encode_new_artist(new_artist, scaler, mlb)

    original_num_features = data.x.shape[1]
    if new_x_np.shape[1] != original_num_features:
        raise ValueError(
            f"Feature mismatch: new artist has {new_x_np.shape[1]} features, "
            f"but model expects {original_num_features}."
        )

    new_x = torch.tensor(new_x_np, dtype=torch.float)

    # ---- Step 2: append new node to feature matrix ----
    updated_x = torch.cat([data.x.cpu(), new_x], dim=0)

    new_node_idx = updated_x.shape[0] - 1

    # ---- Step 3: find collaborator indices ----
    name_to_idx = {name: idx for idx, name in enumerate(filtered_nodes["name"])}

    collaborator_indices = []
    missing_names = []

    for name in collaborator_names:
        if name in name_to_idx:
            collaborator_indices.append(name_to_idx[name])
        else:
            missing_names.append(name)

    if missing_names:
        print("Warning: these collaborators were not found in filtered_nodes:")
        for name in missing_names:
            print(" -", name)

    if len(collaborator_indices) == 0:
        raise ValueError("No valid collaborators found. Please choose names that exist in filtered_nodes.")

    # ---- Step 4: append edges in both directions ----
    new_edges = []
    for idx in collaborator_indices:
        new_edges.append([new_node_idx, idx])
        new_edges.append([idx, new_node_idx])

    new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()

    updated_edge_index = torch.cat([data.edge_index.cpu(), new_edges], dim=1)

    # ---- Step 5: extend labels and masks ----
    # label is unknown, so use placeholder 0
    updated_y = torch.cat([data.y.cpu(), torch.tensor([0], dtype=torch.long)], dim=0)

    updated_train_mask = torch.cat([data.train_mask.cpu(), torch.tensor([False])], dim=0)
    updated_val_mask = torch.cat([data.val_mask.cpu(), torch.tensor([False])], dim=0)
    updated_test_mask = torch.cat([data.test_mask.cpu(), torch.tensor([False])], dim=0)

    # ---- Step 6: build updated Data object ----
    updated_data = Data(
        x=updated_x,
        edge_index=updated_edge_index,
        y=updated_y,
        train_mask=updated_train_mask,
        val_mask=updated_val_mask,
        test_mask=updated_test_mask,
    ).to(device)

    # ---- Step 7: run inference ----
    with torch.no_grad():
        logits, embeddings = model(updated_data.x, updated_data.edge_index)

        new_artist_logits = logits[new_node_idx]
        new_artist_probs = F.softmax(new_artist_logits, dim=0)
        predicted_class = int(torch.argmax(new_artist_probs).item())
        high_popularity_prob = float(new_artist_probs[1].item())

    # ---- Step 8: compare new embedding against original artists only ----
    all_embeddings = embeddings.cpu().numpy()
    original_embeddings = all_embeddings[:-1]   # exclude the new node
    new_embedding = all_embeddings[-1].reshape(1, -1)

    nbrs = NearestNeighbors(n_neighbors=min(top_k, len(original_embeddings)), metric="cosine")
    nbrs.fit(original_embeddings)

    distances, indices = nbrs.kneighbors(new_embedding)

    similar_artists = []
    for idx in indices[0]:
        similar_artists.append(filtered_nodes.iloc[idx]["name"])

    # ---- Step 9: print summary ----
    print("\n--- New Artist Inference ---")
    print("New artist:", new_artist.get("name", "Unknown"))
    print("Collaborators used:", collaborator_names)
    print("Predicted class:", predicted_class)
    print("Prediction meaning:", "High popularity" if predicted_class == 1 else "Not high popularity")
    print(f"Probability of high popularity: {high_popularity_prob:.4f}")

    print("\nNearest existing artists in embedding space:")
    for artist in similar_artists:
        print(" -", artist)

    return {
        "predicted_class": predicted_class,
        "high_popularity_probability": high_popularity_prob,
        "similar_artists": similar_artists,
        "new_embedding": new_embedding,
        "updated_data": updated_data,
    }

def plot_artist_embeddings(embeddings, nodes_df, y):
    """
    Plot learned GNN embeddings in 2D using PCA.
    Color shows popularity class.
    """
    # Reduce embeddings from 64 dimensions down to 2 dimensions
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    # y = 0 means not high popularity
    # y = 1 means high popularity
    plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=y,
        s=20,
        alpha=0.7
    )

    plt.title("GNN Artist Embeddings")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.colorbar(label="Popularity Class")
    plt.show()




def main():
    edges_df, filtered_nodes = load_and_filter_data()

    # Reset index so row positions match tensor rows
    filtered_nodes = filtered_nodes.reset_index(drop=True)

    x, y, mlb, scaler = prepare_node_features(filtered_nodes)
    edge_index, id_to_idx = build_graph(edges_df, filtered_nodes)
    data = create_data_object(x, y, edge_index)
    filtered_nodes["chart_hits_count"] = filtered_nodes["chart_hits"].apply(count_chart_hits)

    high_pop = filtered_nodes[filtered_nodes["popularity"] >= 70]
    low_pop = filtered_nodes[filtered_nodes["popularity"] < 70]

    print("\nHigh popularity artist stats:")
    print(high_pop[["followers", "chart_hits_count", "popularity"]].describe())

    print("\nLow popularity artist stats:")
    print(low_pop[["followers", "chart_hits_count", "popularity"]].describe())

    print("\nData object:")
    print(data)
    print("Feature shape:", data.x.shape)
    print("Edge index shape:", data.edge_index.shape)

    model, device = train_model(data)
    embeddings, preds = evaluate_model(model, data, device)

    # Visual representation of the learned embeddings.
    plot_artist_embeddings(embeddings, filtered_nodes, y)

    scenarios = [
    {
        "name": "Low Feature Artist",
        "artist": {
            "name": "Low Feature Artist",
            "followers": 5000,
            "chart_hits_count": 0,
            "genres": ["hip hop", "rap"]
        },
        "collaborators": ["Bobby Creekwater"]
    },
    {
        "name": "Medium Feature Artist",
        "artist": {
            "name": "Medium Feature Artist",
            "followers": 120000,
            "chart_hits_count": 2,
            "genres": ["hip hop", "rap"]
        },
        "collaborators": ["Drake", "Future"]
    },
    {
        "name": "High Feature Artist",
        "artist": {
            "name": "High Feature Artist",
            "followers": 5000000,
            "chart_hits_count": 10,
            "genres": ["hip hop", "rap", "pop rap"]
        },
        "collaborators": ["Drake", "Future", "Kendrick Lamar", "J. Cole"]
    }
        ]

    for scenario in scenarios:
        print("\n==============================")
        print("Scenario:", scenario["name"])
        predict_new_artist_popularity(
            model=model,
            data=data,
            filtered_nodes=filtered_nodes,
            scaler=scaler,
            mlb=mlb,
            new_artist=scenario["artist"],
            collaborator_names=scenario["collaborators"],
            device=device,
            top_k=5
        )
if __name__ == "__main__":
    main()