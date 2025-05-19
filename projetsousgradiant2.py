import numpy as np
import networkx as nx

def saisir_matrice_distances():
    while True:
        try:
            n = int(input("Entrez le nombre de sommets (>= 3) : "))
            if n < 3:
                print("Le nombre de sommets doit être au moins 3.")
                continue
            break
        except ValueError:
            print("Entrée invalide, veuillez entrer un entier.")
    dist = np.zeros((n, n))
    print("Entrez les distances entre chaque paire de sommets :")
    for i in range(n):
        for j in range(i+1, n):
            while True:
                try:
                    val = float(input(f"Distance entre sommet {i} et {j} : "))
                    if val <= 0:
                        print("La distance doit être strictement positive.")
                        continue
                    dist[i, j] = val
                    dist[j, i] = val
                    break
                except ValueError:
                    print("Entrée invalide, veuillez entrer un nombre.")
    return dist

def calcul_cout_cycle(chemin, dist):
    cout = 0
    for i in range(len(chemin)-1):
        cout += dist[chemin[i], chemin[i+1]]
    return cout

def amelioration_2opt(chemin, dist):
    n = len(chemin) - 1  # chemin fermé, dernier = premier
    amelioration = True
    while amelioration:
        amelioration = False
        for i in range(1, n-1):
            for j in range(i+1, n):
                if j - i == 1:
                    continue
                delta = (dist[chemin[i-1], chemin[j]] + dist[chemin[i], chemin[(j+1)%n]]) - (dist[chemin[i-1], chemin[i]] + dist[chemin[j], chemin[(j+1)%n]])
                if delta < -1e-12:
                    chemin[i:j+1] = reversed(chemin[i:j+1])
                    amelioration = True
        if amelioration:
            continue
    return chemin

def plus_proche_voisin(dist, depart=0):
    n = len(dist)
    visite = [False]*n
    visite[depart] = True
    chemin = [depart]
    actuel = depart
    for _ in range(n-1):
        candidats = [dist[actuel][j] if not visite[j] else np.inf for j in range(n)]
        suivant = np.argmin(candidats)
        visite[suivant] = True
        chemin.append(suivant)
        actuel = suivant
    chemin.append(depart)
    cout = calcul_cout_cycle(chemin, dist)
    return chemin, cout

def construire_1_arbre(dist_modifiee, racine=0):
    n = dist_modifiee.shape[0]
    V = set(range(n))
    V_ = V - {racine}
    G = nx.Graph()
    for i in V_:
        for j in V_:
            if i < j:
                G.add_edge(i, j, weight=dist_modifiee[i, j])
    mst = nx.minimum_spanning_tree(G, weight='weight')
    edges_racine = [(racine, j, dist_modifiee[racine,j]) for j in V_]
    edges_racine.sort(key=lambda x: x[2])
    deux_plus_petites = edges_racine[:2]
    one_tree = nx.Graph()
    one_tree.add_edges_from(mst.edges(data=True))
    for (u,v,w) in deux_plus_petites:
        one_tree.add_edge(u,v, weight=w)
    return one_tree

def degre_noeuds(graphe, n):
    degres = [0]*n
    for node in range(n):
        degres[node] = graphe.degree(node)
    return degres

def longueur_1_arbre(one_tree):
    return sum(attr['weight'] for u,v,attr in one_tree.edges(data=True))

def sous_gradient_tsp(dist, alpha_init=1.5, epsilon=1e-7, kmax=500, racine=0):
    n = dist.shape[0]
    lambdas = np.zeros(n)
    chemin_init, UB = plus_proche_voisin(dist, racine)
    chemin_init = amelioration_2opt(chemin_init, dist)
    UB = calcul_cout_cycle(chemin_init, dist)
    print(f"Borne supérieure initiale améliorée (2-opt) : {UB:.6f}")
    k = 0
    LB = -np.inf
    alpha = alpha_init

    while k < kmax:
        dist_modifiee = np.copy(dist)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_modifiee[i,j] = dist[i,j] - lambdas[j]

        one_tree = construire_1_arbre(dist_modifiee, racine)
        longueur = longueur_1_arbre(one_tree)
        LBk = longueur + lambdas.sum()

        if LBk > LB:
            LB = LBk

        degres = degre_noeuds(one_tree, n)
        g = np.array([2 - deg for deg in degres])
        norm_g = np.linalg.norm(g)

        if norm_g == 0:
            print("Sous-gradient nul, solution optimale trouvée.")
            break

        tk = alpha * (UB - LBk) / (norm_g**2)
        if tk <= 0:
            tk = 1e-6

        lambdas = np.maximum(0, lambdas + tk * g)

        chemin_pv, _ = plus_proche_voisin(dist, racine)
        chemin_opt = amelioration_2opt(chemin_pv, dist)
        UB_candidate = calcul_cout_cycle(chemin_opt, dist)
        if UB_candidate < UB:
            UB = UB_candidate

        ecart_relatif = (UB - LBk) / UB

        print(f"Iter {k+1:3d} | LB = {LBk:.6f} | UB = {UB:.6f} | Ecart relatif = {ecart_relatif:.10f} | ||g|| = {norm_g:.6f} | tk = {tk:.8f} | alpha = {alpha:.5f}")

        # Pour forcer plus d'itérations, commente temporairement les break suivants
        if ecart_relatif < epsilon:
            print("Convergence atteinte avec un écart relatif faible.")
            break

        if norm_g < 1e-8:
            print("Sous-gradient proche de zéro, arrêt.")
            break

        k += 1
        alpha *= 0.99  # décroissance plus lente

    return {
        "lambdas": lambdas,
        "borne_inférieure": LB,
        "borne_supérieure": UB,
        "iterations": k
    }


def main():
    dist = saisir_matrice_distances()
    resultats = sous_gradient_tsp(dist)
    print("\nRésultats finaux :")
    print("Multiplicateurs lambda :", resultats['lambdas'])
    print("Borne inférieure :", resultats['borne_inférieure'])
    print("Borne supérieure :", resultats['borne_supérieure'])
    print("Nombre d'itérations :", resultats['iterations'])

if __name__ == "__main__":
    main()
