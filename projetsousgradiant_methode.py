import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def saisir_matrice():
    nombre_sommets = int(input("Nombre de sommets ? "))
    distances = np.zeros((nombre_sommets, nombre_sommets))
    print("Entrez les distances entre chaque paire de sommets (distances positives).")
    for i in range(nombre_sommets):
        for j in range(i + 1, nombre_sommets):
            while True:
                try:
                    valeur = float(input(f"Distance entre sommet {i} et {j} : "))
                    if valeur < 0:
                        print("Distance positive requise.")
                        continue
                    distances[i, j] = valeur
                    distances[j, i] = valeur
                    break
                except ValueError:
                    print("Valeur invalide, recommencez.")
    return distances

def arbre_couvrant_minimum(dist_modifiees):
    n = dist_modifiees.shape[0]
    graphe = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            graphe.add_edge(i, j, weight=dist_modifiees[i, j])
    mst = nx.minimum_spanning_tree(graphe, weight='weight')
    return mst

def calcul_degres(mst, n):
    degres = np.zeros(n)
    for noeud in mst.nodes():
        degres[noeud] = mst.degree(noeud)
    return degres

def extraire_chemins(mst, n):
    chemins = []
    for depart in range(n):
        chemin = []
        visites = set()
        def parcours_profondeur(u):
            visites.add(u)
            chemin.append(u)
            for voisin in sorted(mst.neighbors(u)):
                if voisin not in visites:
                    parcours_profondeur(voisin)
        parcours_profondeur(depart)
        chemins.append(chemin)
    return chemins

def calcul_cout(distances, chemin):
    cout = 0
    for i in range(len(chemin)-1):
        cout += distances[chemin[i], chemin[i+1]]
    cout += distances[chemin[-1], chemin[0]]
    return cout

def two_opt(chemin, distances):
    n = len(chemin)
    amelioration = True
    while amelioration:
        amelioration = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                nouveau_chemin = chemin[:i] + chemin[i:j][::-1] + chemin[j:]
                if calcul_cout(distances, nouveau_chemin) < calcul_cout(distances, chemin):
                    chemin = nouveau_chemin
                    amelioration = True
    return chemin

def methode_sous_gradient(distances, iterations_max=200, alpha_initial=2.0, tolerance=1e-5):
    n = distances.shape[0]
    multiplicateurs = np.zeros(n)
    meilleur_primal = float('inf')
    meilleur_dual = -float('inf')
    meilleur_chemin = None
    historique_primal = []
    historique_dual = []
    derniers_chemins = []

    for iteration in range(1, iterations_max + 1):
        alpha = alpha_initial / np.sqrt(iteration)
        dist_modifiees = distances - multiplicateurs[:, None] - multiplicateurs[None, :]
        mst = arbre_couvrant_minimum(dist_modifiees)
        degres = calcul_degres(mst, n)
        sous_gradient = degres - 2
        norme_sg = np.linalg.norm(sous_gradient)
        if norme_sg < tolerance:
            print(f"Convergence (sous-gradient faible) à l'itération {iteration}")
            break
        somme_arcs = sum(distances[i][j] for i, j in mst.edges())
        borne_duale = somme_arcs + 2 * np.sum(multiplicateurs)
        if borne_duale > meilleur_dual:
            meilleur_dual = borne_duale
        chemins = extraire_chemins(mst, n)
        derniers_chemins = chemins
        meilleur_cout_iteration = float('inf')
        meilleur_chemin_iteration = None
        for chemin in chemins:
            chemin_opt = two_opt(chemin, distances)
            cout = calcul_cout(distances, chemin_opt)
            if cout < meilleur_cout_iteration:
                meilleur_cout_iteration = cout
                meilleur_chemin_iteration = chemin_opt
        historique_primal.append(meilleur_cout_iteration)
        historique_dual.append(borne_duale)
        if meilleur_cout_iteration < meilleur_primal:
            meilleur_primal = meilleur_cout_iteration
            meilleur_chemin = meilleur_chemin_iteration
        pas = alpha * (meilleur_primal - borne_duale) / (norme_sg**2 + 1e-10)
        if pas < 0:
            pas = 0.1
        multiplicateurs = multiplicateurs + pas * sous_gradient
        print(f"Iter {iteration:3d} | alpha: {alpha:.4f} | Borne duale: {borne_duale:.4f} | Coût primal: {meilleur_cout_iteration:.4f} | Norme SG: {norme_sg:.4f} | Pas: {pas:.4f}")

    return meilleur_chemin, meilleur_primal, historique_primal, historique_dual, derniers_chemins

def afficher_chemins(distances, chemins, meilleur_chemin):
    n = distances.shape[0]
    graphe = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            graphe.add_edge(i, j, weight=distances[i, j])
    position = nx.circular_layout(graphe)
    plt.figure(figsize=(10,10))
    nx.draw(graphe, position, node_color='lightblue', with_labels=True, node_size=500)
    for chemin in chemins:
        arretes = [(chemin[i], chemin[i+1]) for i in range(len(chemin)-1)]
        arretes.append((chemin[-1], chemin[0]))
        nx.draw_networkx_edges(graphe, position, edgelist=arretes, edge_color='gray', style='dashed', width=1, alpha=0.3)
    arretes_meilleur = [(meilleur_chemin[i], meilleur_chemin[i+1]) for i in range(len(meilleur_chemin)-1)]
    arretes_meilleur.append((meilleur_chemin[-1], meilleur_chemin[0]))
    nx.draw_networkx_edges(graphe, position, edgelist=arretes_meilleur, edge_color='red', width=3)
    plt.title("Chemins extraits (gris) et meilleur chemin (rouge)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    distances = saisir_matrice()
    print("\nDémarrage méthode sous-gradient améliorée...\n")
    chemin, cout, hist_p, hist_d, derniers_chemins = methode_sous_gradient(distances, iterations_max=200, alpha_initial=2.0)
    print("\nMeilleur chemin trouvé :", chemin)
    print(f"Coût du meilleur chemin : {cout:.4f}")
    afficher_chemins(distances, derniers_chemins, chemin)
