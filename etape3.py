# ... code existant ...

# Exemple d'utilisation
if __name__ == "__main__":
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
                        val = float(input(f"Distance entre sommet {i+1} et {j+1} : "))
                        if val <= 0:
                            print("La distance doit être strictement positive.")
                            continue
                        dist[i, j] = val
                        dist[j, i] = val
                        break
                    except ValueError:
                        print("Entrée invalide, veuillez entrer un nombre.")
        return dist

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

    def construire_tour_depuis_1arbre(one_tree, dist, racine=0):
        """Construit un tour hamiltonien à partir d'un 1-arbre"""
        # Créer un parcours préfixe de l'arbre
        edges = list(one_tree.edges())
        n = len(dist)
        
        # Créer un graphe eulérien en doublant les arêtes
        euler_graph = nx.MultiGraph()
        for u, v in edges:
            euler_graph.add_edge(u, v, weight=dist[u, v])
            euler_graph.add_edge(u, v, weight=dist[u, v])
        
        # Trouver un circuit eulérien
        try:
            euler_circuit = list(nx.eulerian_circuit(euler_graph, source=racine))
        except nx.NetworkXError:
            # Si le graphe n'est pas eulérien, on utilise l'heuristique du plus proche voisin
            return plus_proche_voisin(dist, racine)[0]
        
        # Extraire un tour hamiltonien en évitant les répétitions
        visite = set()
        tour = []
        
        for u, v in euler_circuit:
            if u not in visite:
                tour.append(u)
                visite.add(u)
        
        # Ajouter les sommets manquants
        for i in range(n):
            if i not in visite:
                tour.append(i)
                visite.add(i)
        
        # Fermer le tour
        tour.append(tour[0])
        
        return tour
    
    def amelioration_2opt(tour, dist):
        """Améliore un tour en utilisant l'heuristique 2-opt"""
        amelioration = True
        meilleur_tour = tour.copy()
        
        while amelioration:
            amelioration = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    # Calculer le coût avant et après l'échange
                    cout_avant = dist[tour[i-1], tour[i]] + dist[tour[j], tour[j+1]]
                    cout_apres = dist[tour[i-1], tour[j]] + dist[tour[i], tour[j+1]]
                    
                    if cout_apres < cout_avant:
                        # Effectuer l'échange 2-opt
                        meilleur_tour[i:j+1] = reversed(tour[i:j+1])
                        amelioration = True
                        tour = meilleur_tour.copy()
                        break
                if amelioration:
                    break
        
        return meilleur_tour
    
    def calculer_cout_tour(tour, dist):
        """Calcule le coût total d'un tour"""
        return sum(dist[tour[i], tour[i+1]] for i in range(len(tour)-1))
    
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
            
            # Étape 1: Construire le 1-arbre
            one_tree = construire_1_arbre(dist_modifiee, racine)
            longueur = longueur_1_arbre(one_tree)
            LBk = longueur + lambdas.sum()
            if LBk > LB:
                LB = LBk
            
            # Étape 2: Améliorer la borne supérieure en construisant un tour à partir du 1-arbre
            tour_candidat = construire_tour_depuis_1arbre(one_tree, dist, racine)
            
            # Étape 3: Appliquer l'amélioration 2-opt
            tour_ameliore = amelioration_2opt(tour_candidat, dist)
            
            # Étape 4: Calculer le coût du tour amélioré
            cout_tour = calculer_cout_tour(tour_ameliore, dist)
            
            # Étape 5: Mettre à jour la borne supérieure si nécessaire
            if cout_tour < UB:
                UB = cout_tour
                print(f"Nouvelle borne supérieure trouvée: {UB:.2f}")
            
            # Continuer avec le reste de l'algorithme du sous-gradient
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
        # Suppression de l'affichage du meilleur chemin

    main()