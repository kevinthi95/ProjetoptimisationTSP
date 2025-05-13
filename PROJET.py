import pulp
import networkx as nx
import matplotlib.pyplot as plt

# Nombre de sommets
n = 5

# Matrice des distances
distance_matrix = [
    [0, 2, 9, 10, 7],
    [1, 0, 6, 4, 3],
    [15, 7, 0, 8, 3],
    [6, 3, 12, 0, 9],
    [10, 4, 8, 6, 0]
]

# Modèle de problème
prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)

# Variables x[i][j]
x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n)] for i in range(n)]

# Variables MTZ
u = [pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n, cat="Integer") for i in range(n)]

# Fonction objectif
prob += pulp.lpSum(distance_matrix[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)

# Contraintes
for i in range(n):
    prob += x[i][i] == 0

for i in range(n):
    prob += pulp.lpSum(x[i][j] for j in range(n) if j != i) == 1
    prob += pulp.lpSum(x[j][i] for j in range(n) if j != i) == 1

prob += u[0] == 1
for i in range(1, n):
    prob += u[i] >= 2
    prob += u[i] <= n

for i in range(n):
    for j in range(n):
        if i != j:
            prob += u[i] - u[j] + n * x[i][j] <= n - 1

# Résolution
solver = pulp.PULP_CBC_CMD(msg=False)
prob.solve(solver)

# Construction du graphe
G = nx.complete_graph(n)
pos = nx.spring_layout(G, seed=42)

# Dessin de tous les arcs
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800)

# Dessin des distances
edge_labels = {(i, j): distance_matrix[i][j] for i in range(n) for j in range(n) if i < j}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Dessin des arcs sélectionnés (x[i][j] == 1)
solution_edges = [(i, j) for i in range(n) for j in range(n) if pulp.value(x[i][j]) == 1]
nx.draw_networkx_edges(G, pos, edgelist=solution_edges, edge_color='red', width=2.5)

plt.title("Chemin optimal du TSP")
plt.show()
