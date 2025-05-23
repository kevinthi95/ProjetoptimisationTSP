import numpy as np
import itertools

# Matrice des distances de notre exemple
D = np.array([
    [0, 12, 9, 24, 18, 14],
    [12, 0, 7, 20, 15, 21],
    [9, 7, 0, 18, 12, 16],
    [24, 20, 18, 0, 10, 14],
    [18, 15, 12, 10, 0, 9],
    [14, 21, 16, 14, 9, 0],
])

def calcul_cout_tour(tour, D):
    cout = 0
    for i in range(len(tour) - 1):
        cout += D[tour[i], tour[i+1]]
    return cout

def tsp_force_brute(D):
    n = D.shape[0]
    villes = list(range(1, n)) 
    cout_min = float('inf')
    meilleur_tour = None

    for perm in itertools.permutations(villes):
        tour = [0] + list(perm) + [0]
        cout = calcul_cout_tour(tour, D)
        if cout < cout_min:
            cout_min = cout
            meilleur_tour = tour

    return meilleur_tour, cout_min

tour_optimal, cout_optimal = tsp_force_brute(D)
print("Tour optimal :", [v+1 for v in tour_optimal])
print("Coût optimal :", cout_optimal)
