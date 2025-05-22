import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt

def generer_matrice_fixe():
    """Génère une matrice de distances fixé par l'utilisateur et reproductible avec des valeurs entières, 
    c'est à dire qu'a chaque fois qu'on demandera un nombre de ville x, on aura toujours la même solution"""
    np.random.seed(45)  # Pour la reproductibilité
    
    # Demander à l'utilisateur la dimension de la matrice
    while True:
        try:
            n = int(input("Entrez la dimension de la matrice (nombre de villes) : "))
            if n < 3:
                print("La dimension doit être au moins 3.")
                continue
            break
        except ValueError:
            print("Entrée invalide, veuillez entrer un entier.")
    
    # Générer des coordonnées de villes dans un plan 
    x = np.random.randint(0, 100, n)
    y = np.random.randint(0, 100, n)
    
    # Calculer les distances euclidiennes et les arrondir à l'entier le plus proche (nous utilisons que des distances entières pour que ce soit plus lisible)
    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = int(np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2))
    
    return dist, x, y

def plus_proche_voisin(dist, depart=0):
    """Méthode du plus proche voisin"""
    n = len(dist)
    visite = [False] * n
    visite[depart] = True
    chemin = [depart]
    cout = 0
    actuel = depart
    
    for _ in range(n-1):
        distances_non_visitees = [dist[actuel][j] if not visite[j] else np.inf for j in range(n)]
        suivant = np.argmin(distances_non_visitees)
        cout += dist[actuel][suivant]
        visite[suivant] = True
        chemin.append(suivant)
        actuel = suivant
    
    cout += dist[actuel][depart]
    chemin.append(depart)
    return chemin, cout

""" Construction du 1-arbre modifié
    - On modifie la matrice des distances en soustrayant les multiplicateurs λ
    - On construit un arbre couvrant minimal (MST) sur les noeuds sauf la racine
    - On ajoute les deux plus petites arêtes incidentes à la racine"""

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
    
    edges_racine = [(racine, j, dist_modifiee[racine, j]) for j in V_]
    edges_racine.sort(key=lambda x: x[2])
    deux_plus_petites = edges_racine[:2]
    
    one_tree = nx.Graph()
    one_tree.add_edges_from(mst.edges(data=True))
    for (u, v, w) in deux_plus_petites:
        one_tree.add_edge(u, v, weight=w)
    
    return one_tree

""" Calcul des degrés des nœuds """

def degre_noeuds(graphe, n):
    degres = [0] * n
    for node in range(n):
        degres[node] = graphe.degree(node)
    return degres

"""Calcul de la distance totale du 1-arbre"""

def longueur_1_arbre(one_tree):
    return sum(attr['weight'] for u, v, attr in one_tree.edges(data=True))

"""Construction d'un tour hamiltonien à partir du 1-arbre"""

def construire_tour_depuis_1arbre(one_tree, dist, racine=0):
    n = len(dist)
    
    if one_tree.number_of_edges() < n - 1:
        chemin, _ = plus_proche_voisin(dist, racine)
        return chemin
    
    # Identifier les nœuds de degré impair
    degrees = dict(one_tree.degree())
    odd_vertices = [v for v, d in degrees.items() if d % 2 == 1]
    
    # Créer un multigraphe
    multigraph = nx.MultiGraph(one_tree)
    
    # Apparier les nœuds de degré impair
    if len(odd_vertices) > 0:
        for i in range(0, len(odd_vertices) - 1, 2):
            u, v = odd_vertices[i], odd_vertices[i + 1]
            multigraph.add_edge(u, v, weight=dist[u, v])
    
    # Trouver un circuit eulérien
    try:
        euler_circuit = list(nx.eulerian_circuit(multigraph, source=racine))
    except:
        chemin, _ = plus_proche_voisin(dist, racine)
        return chemin
    
    # Convertir en tour hamiltonien
    visite = set()
    tour = []
    
    for u, v in euler_circuit:
        if u not in visite:
            tour.append(u)
            visite.add(u)
        if v not in visite:
            tour.append(v)
            visite.add(v)
    
    # Ajouter les nœuds manquants
    for i in range(n):
        if i not in visite:
            tour.append(i)
    
    # Fermer le tour
    if len(tour) > 0 and tour[-1] != tour[0]:
        tour.append(tour[0])
    
    return tour

"""Amélioration locale 2-opt"""

def amelioration_2opt(tour, dist):
    n = len(tour) - 1
    amelioration = True
    meilleur_tour = tour.copy()
    iterations = 0
    max_iterations = n * 10  # Limiter les itérations pour les grandes valeurs afin d'éviter un algo trop long
    
    while amelioration and iterations < max_iterations:
        amelioration = False
        iterations += 1
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cout_avant = dist[meilleur_tour[i-1], meilleur_tour[i]] + dist[meilleur_tour[j], meilleur_tour[j+1]]
                cout_apres = dist[meilleur_tour[i-1], meilleur_tour[j]] + dist[meilleur_tour[i], meilleur_tour[j+1]]
                
                if cout_apres < cout_avant:
                    nouveau_tour = meilleur_tour.copy()
                    nouveau_tour[i:j+1] = reversed(meilleur_tour[i:j+1])
                    meilleur_tour = nouveau_tour
                    amelioration = True
                    break
            if amelioration:
                break
    
    return meilleur_tour

"""Calcul de la longueur d'un tour"""

def calculer_cout_tour(tour, dist):
    return sum(dist[tour[i], tour[i+1]] for i in range(len(tour)-1))

"""Algorithme du sous-gradient """

def sous_gradient_tsp(dist, alpha=2.0, epsilon=0.01, kmax=500, racine=0):
    n = dist.shape[0]
    lambdas = np.zeros(n)
    
    print("Calcul de la solution initiale...")
    chemin_initial, UB = plus_proche_voisin(dist, racine)
    meilleur_tour = chemin_initial
    print(f"Borne supérieure initiale : {UB:.2f}")
    
    k = 0
    LB = -np.inf
    best_LB = -np.inf
    stagnation_count = 0
    alpha_initial = alpha
    
    debut = time.time()
    
    while k < kmax:
        # Modification de la matrice des distances
        dist_modifiee = np.copy(dist)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_modifiee[i, j] = dist[i, j] - lambdas[i] - lambdas[j]
        
        # Construction du 1-arbre
        one_tree = construire_1_arbre(dist_modifiee, racine)
        longueur = longueur_1_arbre(one_tree)
        LBk = longueur + 2 * lambdas.sum()
        
        if LBk > LB:
            LB = LBk
            if LBk > best_LB:
                best_LB = LBk
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1
        
        # Construction et amélioration du tour (moins fréquente pour économiser du temps)
        if k % 5 == 0:
            tour_candidat = construire_tour_depuis_1arbre(one_tree, dist, racine)
            tour_ameliore = amelioration_2opt(tour_candidat, dist)
            cout_tour = calculer_cout_tour(tour_ameliore, dist)
            
            if cout_tour < UB:
                UB = cout_tour
                meilleur_tour = tour_ameliore
                print(f"Iteration {k+1:3d}: Nouvelle borne supérieure = {UB:.2f}")
                stagnation_count = 0
        
        # Calcul du sous-gradient
        degres = degre_noeuds(one_tree, n)
        g = np.array([2 - deg for deg in degres])
        norm_g = np.linalg.norm(g)
        
        if norm_g < 1e-6:
            print("Sous-gradient nul, solution optimale du problème relaxé trouvée.")
            break
        
        # Ajustement adaptatif du pas
        if stagnation_count > 20:
            alpha *= 0.9
            stagnation_count = 0
        
        tk = alpha * (UB - LBk) / (norm_g**2)
        lambdas = np.maximum(0, lambdas + tk * g)
        
        # Affichage périodique pour économiser du temps 
        if k % 25 == 0:
            ecart_relatif = (UB - LBk) / UB * 100 if UB > 0 else 0
            temps_ecoule = time.time() - debut
            print(f"Iter {k+1:3d} | LB={LBk:8.2f} | UB={UB:8.2f} | Ecart={ecart_relatif:5.1f}% | ||g||={norm_g:6.2f} | alpha={alpha:.4f} | Temps={temps_ecoule:6.1f}s")
        
        # Critères d'arrêt
        if (UB - LBk) / UB < epsilon:
            print("Convergence atteinte avec un écart relatif acceptable.")
            break
        
        k += 1
    
    temps_total = time.time() - debut
    
    return {
        "lambdas": lambdas,
        "borne_inférieure": LB,
        "borne_supérieure": UB,
        "meilleur_tour": meilleur_tour,
        "iterations": k,
        "ecart_final": (UB - LB) / UB * 100 if UB > 0 else 0,
        "temps_execution": temps_total
    }


"""Visualise les points et le tour trouvé"""

def visualiser_tour(x, y, tour, titre="Tour optimal", afficher_tous_chemins=False):
    plt.figure(figsize=(10, 8))
    
    # Afficher tous les chemins possibles
    if afficher_tous_chemins:
        n = len(x)
        for i in range(n):
            for j in range(i+1, n):
                plt.plot([x[i], x[j]], [y[i], y[j]], 'black', alpha=0.1, linewidth=0.5, zorder=0)
    
    # Tracer les points
    plt.scatter(x, y, c='blue', s=50, zorder=2)
    
    # Numéroter les points 
    for i in range(len(x)):
        plt.annotate(str(i+1), (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    
    # Tracer le tour
    for i in range(len(tour)-1):
        plt.plot([x[tour[i]], x[tour[i+1]]], [y[tour[i]], y[tour[i+1]]], 'r-', alpha=0.7, zorder=1)
    
    plt.title(titre)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
        
    # Afficher le graphique
    plt.show()

def main():
    print("=== TSP - Méthode du sous-gradient ===\n")
    
    # Génération de la matrice fixe
    print("Génération de la matrice de distances...")
    dist, x, y = generer_matrice_fixe()
    
    print(f"Matrice générée : {dist.shape[0]} villes")
    print(f"Distance minimale : {np.min(dist[dist > 0]):.2f}")
    print(f"Distance maximale : {np.max(dist):.2f}")
    print(f"Distance moyenne : {np.mean(dist[dist > 0]):.2f}")
    print("\nDémarrage de l'algorithme...\n")
    
    # Résolution
    resultats = sous_gradient_tsp(dist)
    
    # Affichage des résultats
    print("\n" + "="*70)
    print("RÉSULTATS FINAUX")
    print("="*70)
    print(f"Borne inférieure finale     : {resultats['borne_inférieure']:10.2f}")
    print(f"Borne supérieure finale     : {resultats['borne_supérieure']:10.2f}")
    print(f"Écart relatif final         : {resultats['ecart_final']:10.2f}%")
    print(f"Nombre d'itérations         : {resultats['iterations']:10d}")
    print(f"Temps d'exécution           : {resultats['temps_execution']:10.1f}s")
    print(f"Longueur du meilleur tour   : {resultats['borne_supérieure']:10.2f}")
    print(f"Nombre de villes dans le tour: {len(resultats['meilleur_tour'])-1:10d}")
    print("="*70)
    
    # Visualisation du tour avec tous les chemins possibles
    visualiser_tour(x, y, resultats['meilleur_tour'], 
                   f"Tour optimal - {dist.shape[0]} villes - Longueur: {resultats['borne_supérieure']:.2f}",
                   afficher_tous_chemins=True)

if __name__ == "__main__":
    main()