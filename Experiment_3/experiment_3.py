
# %% [markdown]
# # Experiment 3: Erdős-Rényi (ER) Random Network Model
# 
# **Aim**: Implementation of The Erdős-Rényi (ER) random network growth model and analysis of its properties.
# 
# **Theory**:
# The G(n, p) model generates a graph of n nodes where each pair of nodes is connected with probability p.
# We will analyze:
# - Degree Distribution
# - Diameter
# - Average Clustering Coefficient
# - Effect of changing N and P
# 
# **Note**: The syllabus mentions "Create a scale-free network using the Erdos-enyi network", which is a contradiction. ER networks have a Poisson degree distribution, not scale-free (power-law). We will implement the ER model as named.

# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
def analyze_er_graph(n, p):
    # Create Graph
    G = nx.erdos_renyi_graph(n, p, seed=42)
    
    # Metrics
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        # If not connected, diameter is infinity, or we take the largest component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
    
    avg_clustering = nx.average_clustering(G)
    avg_degree = np.mean([d for n, d in G.degree()])
    
    return {
        "N": n,
        "P": p,
        "Diameter": diameter,
        "Avg Clustering": avg_clustering,
        "Avg Degree": avg_degree,
        "Graph": G
    }

# %%
# Effect of changing N and P
params = [
    (50, 0.1), (50, 0.3), (50, 0.5),
    (100, 0.1), (100, 0.3), (100, 0.5),
    (200, 0.1)
]

results = []
for n, p in params:
    res = analyze_er_graph(n, p)
    results.append(res)
    
# Display Results Table
df = pd.DataFrame(results).drop(columns=["Graph"])
print("Comparison of Network Properties vary N and P:")
print(df)

# %%
# Visualizing Degree Distribution for one case (N=100, P=0.1)
sample_res = results[3] # N=100, P=0.1
G_sample = sample_res["Graph"]

degrees = [d for n, d in G_sample.degree()]
plt.figure(figsize=(10, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), edgecolor='black', alpha=0.7)
plt.title(f"Degree Distribution (N=100, P=0.1)")
plt.xlabel("Degree")
plt.ylabel("Frequency")

# Network Plot
plt.subplot(1, 2, 2)
pos = nx.spring_layout(G_sample, seed=42)
nx.draw(G_sample, pos, node_size=30, node_color='lightgreen', edge_color='gray', width=0.5)
plt.title("Network Visualization")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Observation**:
# 1. **Average Degree**: Directly proportional to $P \times (N-1)$. As P increases, the graph becomes denser.
# 2. **Diameter**: As P increases (more edges), the diameter decreases because nodes are reached more easily.
# 3. **Clustering Coefficient**: In ER graphs, $C \approx P$. We can observe that the average clustering coefficient is close to the probability P.
# 4. **Degree Distribution**: Resembles a Poisson distribution (bell curve), typical for Random Networks, unlike the Power Law seen in Scale-Free networks.
