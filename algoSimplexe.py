import numpy as np

def simplexe(M):
    m, n = M.shape  # m lignes, n colonnes (n-1 variables + 1 colonne des constantes)

    while np.any(M[m-1, :-1] < -1e-10):  # Tant que des coûts réduits sont négatifs
        # Étape 1 : Choisir la variable entrante (coût réduit le plus négatif)
        i_vEntrante = np.argmin(M[m-1, :-1])

        # Étape 2 : Calcul des rapports pour déterminer la variable sortante
        rapports = []
        for i in range(m - 1):
            denom = M[i, i_vEntrante]
            if denom > 1e-10:  # Éviter divisions par 0 ou valeurs négatives
                rapports.append(M[i, -1] / denom)
            else:
                rapports.append(np.inf)  # Ligne non éligible

        if all(r == np.inf for r in rapports):
            raise ValueError("Problème non borné : aucune variable sortante possible")

        i_vSortante = np.argmin(rapports)

        # Étape 3 : Pivot de Gauss-Jordan
        pivot = M[i_vSortante, i_vEntrante]
        M[i_vSortante, :] /= pivot  # Normaliser la ligne du pivot

        for i in range(m):
            if i != i_vSortante:
                facteur = M[i, i_vEntrante]
                M[i, :] -= facteur * M[i_vSortante, :]

    return M

# Exemple : problème de maximisation avec contraintes sous forme standard
M = np.array([
    [1.0, 1.0, 1.0, 0.0, 4.0],  # x + y + s1 = 4
    [2.0, 1.0, 0.0, 1.0, 5.0],  # 2x + y + s2 = 5
    [-2.0, -3.0, 0.0, 0.0, 0.0] # Max Z = 2x + 3y <=> -2x -3y
])

M_final = simplexe(M)

print("Tableau final du simplexe :")
print(M_final)

n_vars = 2  # Nombre de variables de décision (x, y)

solution = np.zeros(n_vars)
M_base = M_final[:-1, :]  # On enlève la dernière ligne (fonction objectif)

for j in range(n_vars):  # Pour chaque variable x, y, ...
    colonne = M_base[:, j]
    if np.count_nonzero(colonne) == 1 and np.isclose(np.max(colonne), 1):
        i = np.argmax(colonne)
        solution[j] = M_base[i, -1]  # Dernière colonne = valeur

# Affichage de la solution
print("\nValeurs des variables :")
for i, val in enumerate(solution):
    print(f"x{i+1} = {val}")


# Extraction de la solution optimale :
print("\nValeur optimale :", M_final[-1, -1])
