import matplotlib.pyplot as plt
import math
import random

# on deamnde le nombre de sommet ( taille matrice )
n = int(input("Combien de sommets ? "))

# on entre chaque distance ( la moitié de la mtrice symétrique )
print("\nEntrez les distances entre chaque paire de sommets (i < j).")
dist_matrix = [[0] * n for _ in range(n)]
for i in range(n):
    for j in range(i + 1, n):
        while True:
            try:
                val = float(input(f"Distance entre sommet {i+1} et sommet {j+1} : "))
                dist_matrix[i][j] = val
                dist_matrix[j][i] = val
                break
            except ValueError:
                print("❌ Entrée invalide. Veuillez entrer un nombre.")

# affichage
coords = [[math.cos(2 * math.pi * i / n) * 10,
           math.sin(2 * math.pi * i / n) * 10] for i in range(n)]

# chemain initiale test
initial_path = list(range(n))
random.shuffle(initial_path)
initial_path.append(initial_path[0])  # Retour au point de départ

# calcule distance total
def calculate_total_distance(path, dist_matrix):
    return sum(dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

# algorithme 2-opt
def two_opt(path, dist_matrix):
    best = path
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                if j - i == 1:
                    continue  # pas d'échange consécutif
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                if calculate_total_distance(new_path, dist_matrix) < calculate_total_distance(best, dist_matrix):
                    best = new_path
                    improved = True
        path = best
    return path

# Application 2-opt
optimized_path = two_opt(initial_path, dist_matrix)
optimized_distance = calculate_total_distance(optimized_path, dist_matrix)

# Affichage du graphe avec les différents chemins
def plot_two_opt_path(path, coords, dist_matrix):
    plt.figure(figsize=(10, 8))

    # Graphe complet
    for i in range(n):
        for j in range(n):
            if i != j:
                xi, yi = coords[i]
                xj, yj = coords[j]
                plt.plot([xi, xj], [yi, yj], color='lightgray', linestyle='--', linewidth=0.8)
                xm, ym = (xi + xj) / 2, (yi + yj) / 2
                plt.text(xm, ym, str(dist_matrix[i][j]), fontsize=7, color='gray', ha='center')

    # Chemin optimisé
    for i in range(len(path) - 1):
        a, b = coords[path[i]], coords[path[i + 1]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=2.5)
        plt.plot([a[0], b[0]], [a[1], b[1]], 'ro')

    # Numérotation des sommets
    for idx, (xpt, ypt) in enumerate(coords):
        plt.text(xpt + 0.5, ypt + 0.5, f"{idx + 1}", fontsize=12, weight='bold', color='black')

    plt.title("Graphe complet avec distances + chemin optimisé (2-opt)", fontsize=14)
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# Tracer le résultat
plot_two_opt_path(optimized_path, coords, dist_matrix)

# Affichage du chemin et de la distance
path_human = [i + 1 for i in optimized_path]
print("\n✅ Résultat")
print("Chemin optimisé (2-opt) :", path_human)
print("Distance totale :", round(optimized_distance, 2))
