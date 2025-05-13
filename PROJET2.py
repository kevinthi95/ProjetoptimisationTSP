import pulp
import networkx as nx
import matplotlib.pyplot as plt

# === Étape 1 : Entrée utilisateur ===
n = int(input("Nombre de sommets (≥ 3) : "))
distance_matrix = [[0 for _ in range(n)] for _ in range(n)]

print("\nEntrez les distances entre chaque paire de sommets.")
print("Pour i ≠ j, entrez la distance entre i et j (entier positif).")

for i in range(n):
    for j in range(n):
        if i != j:
            d = input(f"Distance de {i} à {j} : ")
            while not d.isdigit():
                d = input(f"Distance invalide. Distance de {i} à {j} : ")
            distance_matrix[i][j] = int(d)

# === Étape 2 : Modèle PuLP ===
prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)

x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n)] for i in range(n)]
u = [pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n, cat="Integer") for i in range(n)]

prob += pulp.lpSum(distance_matrix[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)

for i in range(n):
    prob += x[i][i] == 0
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

# === Étape 3 : Résolution ===
solver = pulp.PULP_CBC_CMD(msg=False)
prob.solve(solver)

# === Étape 4 : Résultat ===
status = pulp.LpStatus[prob.status]
print(f"\nRésultat : {status}")
print("Chemin choisi :")
solution_edges = []
for i in range(n):
    for j in range(n):
        if pulp.value(x[i][j]) == 1:
            solution_edges.append((i, j))
            print(f"{i} → {j}")

# === Étape 5 : Affichage graphique ===
G = nx.complete_graph(n)
pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800)
edge_labels = {(i, j): distance_matrix[i][j] for i in range(n) for j in range(n) if i < j}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
nx.draw_networkx_edges(G, pos, edgelist=solution_edges, edge_color='red', width=2.5)

plt.title("Chemin optimal du TSP")
plt.show()
