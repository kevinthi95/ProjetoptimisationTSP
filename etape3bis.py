
if __name__ == "__main__":
    import numpy as np
    import networkx as nx

    """Récupérer la matrice de distance de notre TSP"""

    def saisir_matrice_distances():
        # Utilisation de la matrice prédéfinie au lieu de demander à l'utilisateur
        dist = np.array([
            [ 0, 32, 41, 22, 13, 17, 49, 11, 23, 38, 14, 29, 31, 25, 42],
            [32,  0, 19, 35, 21, 33, 25, 30, 16, 44, 29, 23, 38, 17, 24],
            [41, 19,  0, 20, 31, 27, 15, 18, 39, 22, 24, 42, 19, 26, 20],
            [22, 35, 20,  0, 18, 21, 44, 27, 12, 19, 35, 15, 24, 31, 29],
            [13, 21, 31, 18,  0, 14, 37, 20, 28, 11, 19, 32, 25, 18, 21],
            [17, 33, 27, 21, 14,  0, 29, 13, 36, 22, 27, 35, 20, 22, 19],
            [49, 25, 15, 44, 37, 29,  0, 24, 33, 27, 38, 18, 29, 14, 17],
            [11, 30, 18, 27, 20, 13, 24,  0, 25, 19, 23, 27, 15, 26, 21],
            [23, 16, 39, 12, 28, 36, 33, 25,  0, 11, 20, 14, 27, 18, 24],
            [38, 44, 22, 19, 11, 22, 27, 19, 11,  0, 26, 18, 22, 24, 13],
            [14, 29, 24, 35, 19, 27, 38, 23, 20, 26,  0, 21, 25, 13, 16],
            [29, 23, 42, 15, 32, 35, 18, 27, 14, 18, 21,  0, 17, 29, 20],
            [31, 38, 19, 24, 25, 20, 29, 15, 27, 22, 25, 17,  0, 16, 22],
            [25, 17, 26, 31, 18, 22, 14, 26, 18, 24, 13, 29, 16,  0, 19],
            [42, 24, 20, 29, 21, 19, 17, 21, 24, 13, 16, 20, 22, 19,  0]
        ])
        print("Utilisation d'une matrice de distances 15×15 prédéfinie")
        return dist

    """ Fonction heuristique simple pour initialiser UB """

    def plus_proche_voisin(dist, depart=0):
        n = len(dist)
        visite = [False]*n
        visite[depart] = True
        chemin = [depart]
        cout = 0
        actuel = depart
        for _ in range(n-1):
            suivant = np.argmin([dist[actuel][j] if not visite[j] else np.inf for j in range(n)])
            cout += dist[actuel][suivant]
            visite[suivant] = True
            chemin.append(suivant)
            actuel = suivant
        cout += dist[actuel][depart]
        chemin.append(depart)
        return chemin, cout


    """Étape 1 : Construction du 1-arbre modifié
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
        edges_racine = [(racine, j, dist_modifiee[racine,j]) for j in V_]
        edges_racine.sort(key=lambda x: x[2])
        deux_plus_petites = edges_racine[:2]
        one_tree = nx.Graph()
        one_tree.add_edges_from(mst.edges(data=True))
        for (u,v,w) in deux_plus_petites:
            one_tree.add_edge(u,v, weight=w)
        return one_tree

    """ Étape 2 : Calcul des degrés des noeuds dans le 1-arbre
    - Permet de calculer le sous-gradient"""

    def degre_noeuds(graphe, n):
        degres = [0]*n
        for node in range(n):
            degres[node] = graphe.degree(node)
        return degres

    """ Calcul de la distance totale du 1-arbre """
    def longueur_1_arbre(one_tree):
        return sum(attr['weight'] for u,v,attr in one_tree.edges(data=True))

    """ Étape 5 : Construction heuristique d'un tour valide à partir du 1-arbre
    - Cette fonction crée un chemin qui visite tous les noeuds (tour Hamiltonien)
    - Ici simplifié : parcours naïf en suivant les connexions """

    def construire_tour_depuis_1arbre(one_tree, dist, racine=0):
        edges = list(one_tree.edges())
        n = len(dist)
        
        euler_graph = nx.MultiGraph()
        for u, v in edges:
            euler_graph.add_edge(u, v, weight=dist[u, v])
            euler_graph.add_edge(u, v, weight=dist[u, v])
        
        try:
            euler_circuit = list(nx.eulerian_circuit(euler_graph, source=racine))
        except nx.NetworkXError:
            return plus_proche_voisin(dist, racine)[0]
        
        visite = set()
        tour = []
        
        for u, v in euler_circuit:
            if u not in visite:
                tour.append(u)
                visite.add(u)
        
        for i in range(n):
            if i not in visite:
                tour.append(i)
                visite.add(i)
        
        tour.append(tour[0])
        
        return tour
    
    """ Amélioration locale 2-opt pour réduire le coût du tour """

    def amelioration_2opt(tour, dist):
        amelioration = True
        meilleur_tour = tour.copy()
        
        while amelioration:
            amelioration = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    cout_avant = dist[tour[i-1], tour[i]] + dist[tour[j], tour[j+1]]
                    cout_apres = dist[tour[i-1], tour[j]] + dist[tour[i], tour[j+1]]
                    
                    if cout_apres < cout_avant:
                        meilleur_tour[i:j+1] = reversed(tour[i:j+1])
                        amelioration = True
                        tour = meilleur_tour.copy()
                        break
                if amelioration:
                    break
        
        return meilleur_tour
    
    """ Calcul da longeur d'un tour """

    def calculer_cout_tour(tour, dist):
        return sum(dist[tour[i], tour[i+1]] for i in range(len(tour)-1))
    

    """ Algorithme du sous-gradient pour résoudre le TSP """
    
    def sous_gradient_tsp(dist, alpha=1, epsilon=0.001, kmax=10000, racine=0):
        n = dist.shape[0]
        lambdas = np.zeros(n)
        chemin_initial, UB = plus_proche_voisin(dist, racine)
        print(f"Borne supérieure initiale (heuristique plus proche voisin) : {UB:.2f}")
        
        k = 0
        LB = -np.inf
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
            
            tour_candidat = construire_tour_depuis_1arbre(one_tree, dist, racine)
            
            tour_ameliore = amelioration_2opt(tour_candidat, dist)
            
            cout_tour = calculer_cout_tour(tour_ameliore, dist)
            
            if cout_tour < UB:
                UB = cout_tour
                print(f"Nouvelle borne supérieure trouvée: {UB:.2f}")
            
            degres = degre_noeuds(one_tree, n)
            g = np.array([2 - deg for deg in degres])
            norm_g = np.linalg.norm(g)
            if norm_g == 0:
                print("Sous-gradient nul, solution optimale trouvée.")
                break
            
            tk = alpha * (UB - LBk) / (norm_g**2)
            lambdas = np.maximum(0, lambdas + tk * g)
            
            print(f"Iteration {k+1} | LB={LBk:.2f} | UB={UB:.2f} | Ecart relatif={(UB-LBk)/UB*100:.2f}% | ||g||={norm_g:.2f}")
            
            if (UB - LBk) / UB < epsilon:
                print("Convergence atteinte avec un écart relatif faible.")
                break
            if norm_g < 1e-5:
                print("Sous-gradient proche de zéro, arrêt.")
                break
            
            k += 1
        
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

    main()