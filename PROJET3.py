import numpy as np
import itertools
import time
import matplotlib.pyplot as plt

# === Paramètres ===
np.random.seed(0)
n = 6  # nombre de villes

# === Génération aléatoire des villes ===
coords = np.random.rand(n, 2) * 100
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

# === Fonctions utiles ===
def calculate_total_distance(route, dist_matrix):
    return sum(dist_matrix[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))

def plot_tsp(coords, route, title, color):
    plt.figure()
    x = [coords[i][0] for i in route]
    y = [coords[i][1] for i in route]
    plt.plot(x, y, marker='o', linestyle='-', color=color)
    for idx, (x_i, y_i) in enumerate(coords):
        plt.text(x_i + 1, y_i + 1, str(idx), fontsize=12)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# === 2-opt ===
def two_opt(route, dist_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if calculate_total_distance(new_route, dist_matrix) < calculate_total_distance(best, dist_matrix):
                    best = new_route
                    improved = True
        route = best
    return best

# === Held-Karp ===
def held_karp(dist_matrix):
    n = len(dist_matrix)
    C = {}

    for k in range(1, n):
        C[(1 << k, k)] = (dist_matrix[0][k], 0)

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    res.append((C[(prev, m)][0] + dist_matrix[m][k], m))
                C[(bits, k)] = min(res)

    bits = (2 ** n - 1) - 1
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dist_matrix[k][0], k))
    opt, parent = min(res)

    path = []
    last = parent
    bitmask = bits
    for _ in range(n - 1):
        path.append(last)
        new_bitmask = bitmask & ~(1 << last)
        _, last = C[(bitmask, last)]
        bitmask = new_bitmask
    path.append(0)
    path = path[::-1]
    path.append(0)
    return path, opt

# === Exécution des deux algorithmes ===

# 2-opt
start_2opt = time.time()
route_2opt = two_opt(list(range(n)), distance_matrix)
dist_2opt = calculate_total_distance(route_2opt + [route_2opt[0]], distance_matrix)
time_2opt = time.time() - start_2opt
route_2opt.append(route_2opt[0])

# Held-Karp
start_hk = time.time()
route_hk, dist_hk = held_karp(distance_matrix)
time_hk = time.time() - start_hk
route_hk.append(0)

# === Affichage résultats ===
print("\n===== COMPARAISON DES ALGOS TSP =====\n")
print("🔁 2-opt")
print("Route :", route_2opt)
print("Distance :", round(dist_2opt, 2))
print("Temps :", round(time_2opt, 4), "s\n")

print("🧠 Held-Karp")
print("Route :", route_hk)
print("Distance :", round(dist_hk, 2))
print("Temps :", round(time_hk, 4), "s")

# === Visualisation des parcours ===
plot_tsp(coords, route_2opt, "Parcours 2-opt", "blue")
plot_tsp(coords, route_hk, "Parcours Held-Karp", "green")
