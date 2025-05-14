# Code interactif pour saisir manuellement une matrice de distances et résoudre le TSP avec MTZ

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpInteger, LpStatus, value
import matplotlib.pyplot as plt
import math

# 1. Entrée du nombre de sommets
n = int(input("Combien de sommets ? "))

# 2. Saisie manuelle des distances (matrice symétrique)
print("\nEntrez les distances entre chaque paire de sommets (la diagonale est à 0 automatiquement).")
print("Vous n'entrez que les distances supérieures à la diagonale (i < j), elles seront recopiées en symétrie.")

dist_matrix = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        while True:
            try:
                val = float(input(f"Distance entre sommet {i+1} et sommet {j+1} : "))
                dist_matrix[i][j] = val
                dist_matrix[j][i] = val
                break
            except ValueError:
                print("❌ Entrée invalide. Veuillez entrer un nombre.")

# 3. Coordonnées fictives pour affichage (disposition circulaire)
coords = [[math.cos(2 * math.pi * i / n) * 10, math.sin(2 * math.pi * i / n) * 10] for i in range(n)]

# 4. Résolution du TSP avec formulation MTZ
model = LpProblem("TSP_MTZ", LpMinimize)
x = {(i, j): LpVariable(f"x_{i+1}_{j+1}", cat=LpBinary) for i in range(n) for j in range(n) if i != j}
u = {i: LpVariable(f"u_{i+1}", lowBound=1, upBound=n, cat=LpInteger) for i in range(1, n)}

model += lpSum(dist_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)

for i in range(n):
    model += lpSum(x[i, j] for j in range(n) if i != j) == 1
    model += lpSum(x[j, i] for j in range(n) if i != j) == 1

for i in range(1, n):
    for j in range(1, n):
        if i != j:
            model += u[i] - u[j] + n * x[i, j] <= n - 1

model.solve()

# 5. Extraire le chemin optimal
edges = [(i, j) for i in range(n) for j in range(n) if i != j and x[i, j].varValue == 1]
path = [0]
while len(path) < n:
    for i, j in edges:
        if i == path[-1] and j not in path:
            path.append(j)
            break
path.append(0)

# 6. Affichage graphique
def plot_enhanced_path(path, coords, dist_matrix):
    plt.figure(figsize=(10, 8))
    for i in range(n):
        for j in range(n):
            if i != j:
                xi, yi = coords[i]
                xj, yj = coords[j]
                plt.plot([xi, xj], [yi, yj], color='lightgray', linestyle='--', linewidth=0.8)
                xm, ym = (xi + xj) / 2, (yi + yj) / 2
                plt.text(xm, ym, str(dist_matrix[i][j]), fontsize=7, color='gray', ha='center')

    for i in range(len(path) - 1):
        a, b = coords[path[i]], coords[path[i + 1]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=2.5)
        plt.plot([a[0], b[0]], [a[1], b[1]], 'ro')

    for idx, (xpt, ypt) in enumerate(coords):
        plt.text(xpt + 0.5, ypt + 0.5, f"{idx + 1}", fontsize=12, weight='bold', color='black')

    plt.title("Graphe complet avec distances + chemin optimal (en bleu)", fontsize=14)
    plt.axis("equal")
    plt.grid(True)
    plt.show()

plot_enhanced_path(path, coords, dist_matrix)

# 7. Résultat texte
path_human = [i + 1 for i in path]
print("\n✅ Résultat")
print("Chemin optimal :", path_human)
print("Distance totale :", round(value(model.objective), 2))
print("Statut :", LpStatus[model.status])
