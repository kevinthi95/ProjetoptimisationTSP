from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpInteger, LpStatus, value
import matplotlib.pyplot as plt
import math

# 1. Nombre de sommets
n = int(input("Entrez le nombre de sommets : "))

# 2. Saisie manuelle des distances avec sécurité + symétrie
print("\nEntrez les distances entre chaque paire de sommets (diagonale = 0 automatiquement) :")
dist_matrix = []
for i in range(n):
    row = []
    for j in range(n):
        if i == j:
            row.append(0)
        elif j < i:
            row.append(dist_matrix[j][i])  # copie symétrique
        else:
            while True:
                try:
                    val = float(input(f"Distance entre {i} et {j} : "))
                    row.append(val)
                    break
                except ValueError:
                    print("❌ Entrée invalide. Veuillez entrer un nombre.")
    dist_matrix.append(row)

# 3. Coordonnées pour affichage (position circulaire)
coords = [[math.cos(2 * math.pi * i / n) * 10, math.sin(2 * math.pi * i / n) * 10] for i in range(n)]

# 4. Modèle MTZ avec PuLP
model = LpProblem("TSP_MTZ", LpMinimize)
x = {(i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i in range(n) for j in range(n) if i != j}
u = {i: LpVariable(f"u_{i}", lowBound=1, upBound=n, cat=LpInteger) for i in range(1, n)}

# Fonction objectif
model += lpSum(dist_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)

# Contraintes : chaque sommet a une entrée et une sortie
for i in range(n):
    model += lpSum(x[i, j] for j in range(n) if i != j) == 1
    model += lpSum(x[j, i] for j in range(n) if i != j) == 1

# Contraintes MTZ pour éviter les sous-tours
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            model += u[i] - u[j] + n * x[i, j] <= n - 1

# 5. Résolution
model.solve()

# 6. Reconstruction du chemin optimal
edges = [(i, j) for i in range(n) for j in range(n) if i != j and x[i, j].varValue == 1]
path = [0]
while len(path) < n:
    for i, j in edges:
        if i == path[-1] and j not in path:
            path.append(j)
            break
path.append(0)

# 7. Affichage avec surbrillance du chemin optimal
def plot_path(path, coords):
    plt.figure(figsize=(8, 6))

    # Graphe complet en arrière-plan
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i != j:
                xi, yi = coords[i]
                xj, yj = coords[j]
                plt.plot([xi, xj], [yi, yj], color='lightgray', linestyle='--', linewidth=0.8)

    # Chemin optimal en surbrillance
    for i in range(len(path) - 1):
        a, b = coords[path[i]], coords[path[i+1]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=2.5)
        plt.plot([a[0], b[0]], [a[1], b[1]], 'ro')

    # Numérotation des sommets
    for idx, (xpt, ypt) in enumerate(coords):
        plt.text(xpt + 0.5, ypt + 0.5, str(idx), fontsize=12)

    plt.title("Graphe complet + chemin optimal (surbrillance)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

plot_path(path, coords)

# 8. Résultats affichés
print("\n✅ Résultat")
print("Chemin optimal :", path)
print("Distance totale :", round(value(model.objective), 2))
print("Statut :", LpStatus[model.status])
python6
