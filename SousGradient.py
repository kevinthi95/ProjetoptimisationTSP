if __name__ == "__main__":
    import numpy as np
    import networkx as nx
    import time

    """Récupérer la matrice de distance de notre TSP"""

    def saisir_matrice_distances():
        while True:
            try:
                n = int(input("Entrez le nombre de sommets (>= 3) : "))
                if n < 3:
                    print("Le nombre de sommets doit être au moins 3.")
                    continue
                break
            except ValueError:
                print("Entrée invalide, veuillez entrer une valeure entière.")
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

    """ Méthode plus proche voisin pour initialiser UB """

    def plus_proche_voisin(dist, depart=0): #On part toujours de la ville 1 (indice 0) pour faire le plus proche voisin
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
        edges_racine = [(racine, j, dist_modifiee[racine,j]) for j in V_]
        edges_racine.sort(key=lambda x: x[2])
        deux_plus_petites = edges_racine[:2]
        one_tree = nx.Graph()
        one_tree.add_edges_from(mst.edges(data=True))
        for (u,v,w) in deux_plus_petites:
            one_tree.add_edge(u,v, weight=w)
        return one_tree

    """ Calcul des degrés des noeuds dans le 1-arbre """

    def degre_noeuds(graphe, n):
        degres = [0]*n
        for node in range(n):
            degres[node] = graphe.degree(node)
        return degres

    """ Calcul de la distance totale du 1-arbre """

    def longueur_1_arbre(one_tree):
        return sum(attr['weight'] for u,v,attr in one_tree.edges(data=True))

    """ Construction d'un tour hamiltonien à partir du 1-arbre  """
    
    def construire_tour_depuis_1arbre(one_tree, dist, racine=0):
        n = len(dist)
        
        # Si le 1-arbre n'a pas assez d'arêtes, utiliser l'heuristique du plus proche voisin
        if one_tree.number_of_edges() < n - 1:
            chemin, _ = plus_proche_voisin(dist, racine)
            return chemin
        
        # Créer un graphe avec toutes les arêtes dupliquées pour les nœuds de degré impair
        degrees = dict(one_tree.degree())
        odd_vertices = [v for v, d in degrees.items() if d % 2 == 1]
        
        # Si pas de nœuds de degré impair, c'est déjà eulérien
        multigraph = nx.MultiGraph(one_tree)
        
        # Apparier les nœuds de degré impair et ajouter les arêtes les plus courtes
        if len(odd_vertices) > 0:
            # Méthode simplifiée : connecter les nœuds impairs par paires
            for i in range(0, len(odd_vertices) - 1, 2):
                u, v = odd_vertices[i], odd_vertices[i + 1]
                multigraph.add_edge(u, v, weight=dist[u, v])
        
        # Trouver un circuit eulérien
        try:
            euler_circuit = list(nx.eulerian_circuit(multigraph, source=racine))
        except:
            # En cas d'échec, revenir à l'heuristique du plus proche voisin
            chemin, _ = plus_proche_voisin(dist, racine)
            return chemin
        
        # Convertir le circuit eulérien en tour hamiltonien
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
    
    """ Amélioration locale 2-opt """

    def amelioration_2opt(tour, dist):
        n = len(tour) - 1  # -1 car le tour se termine par le nœud de départ
        amelioration = True
        meilleur_tour = tour.copy()
        
        while amelioration:
            amelioration = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Calculer le coût avant et après l'échange
                    cout_avant = dist[meilleur_tour[i-1], meilleur_tour[i]] + dist[meilleur_tour[j], meilleur_tour[j+1]]
                    cout_apres = dist[meilleur_tour[i-1], meilleur_tour[j]] + dist[meilleur_tour[i], meilleur_tour[j+1]]
                    
                    if cout_apres < cout_avant:
                        # Inverser le segment entre i et j
                        nouveau_tour = meilleur_tour.copy()
                        nouveau_tour[i:j+1] = reversed(meilleur_tour[i:j+1])
                        meilleur_tour = nouveau_tour
                        amelioration = True
                        break
                if amelioration:
                    break
        
        return meilleur_tour
    
    """ Calcul de la longueur d'un tour """

    def calculer_cout_tour(tour, dist):
        return sum(dist[tour[i], tour[i+1]] for i in range(len(tour)-1))
    
    """ Algorithme du sous-gradient """
    
    def sous_gradient_tsp(dist, alpha=2.0, epsilon=0.001, kmax=1000, racine=0):
        n = dist.shape[0]
        lambdas = np.zeros(n)
        chemin_initial, UB = plus_proche_voisin(dist, racine)
        meilleur_tour = chemin_initial
        print(f"Borne supérieure initiale (heuristique plus proche voisin) : {UB:.2f}")
        
        k = 0
        LB = -np.inf
        best_LB = -np.inf
        stagnation_count = 0
        alpha_initial = alpha
        debut = time.time()
        
        #itération tant qu'on dépasse pas kmax
        while k < kmax:
            dist_modifiee = np.copy(dist)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist_modifiee[i,j] = dist[i,j] - lambdas[i] - lambdas[j]
            
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
            
            tour_candidat = construire_tour_depuis_1arbre(one_tree, dist, racine)
            tour_ameliore = amelioration_2opt(tour_candidat, dist)
            cout_tour = calculer_cout_tour(tour_ameliore, dist)
            
            if cout_tour < UB:
                UB = cout_tour
                meilleur_tour = tour_ameliore
                print(f"Nouvelle borne supérieure trouvée: {UB:.2f}")
                stagnation_count = 0
            
            degres = degre_noeuds(one_tree, n)
            g = np.array([2 - deg for deg in degres])
            norm_g = np.linalg.norm(g)
            
            if norm_g < 1e-6:
                print("Sous-gradient nul, solution optimale du problème relaxé trouvée.")
                break
            
            if stagnation_count > 20:
                alpha *= 0.9
                stagnation_count = 0
                print(f"Réduction du pas alpha: {alpha:.4f}")
            
            tk = alpha * (UB - LBk) / (norm_g**2)
            lambdas = np.maximum(0, lambdas + tk * g)
            
            if k % 10 == 0 or k < 20:
                ecart_relatif = (UB - LBk) / UB * 100 if UB > 0 else 0
                temps_ecoule = time.time() - debut
                print(f"Iteration {k+1:4d} | LB={LBk:8.2f} | UB={UB:8.2f} | Ecart={ecart_relatif:6.2f}% | ||g||={norm_g:6.2f} | alpha={alpha:.4f} | Temps={temps_ecoule:6.1f}s")
            
            if (UB - LBk) / UB < epsilon:
                print("Convergence atteinte avec un écart relatif faible.")
                break
            
            k += 1
        
        temps_total = time.time() - debut

        return {
            "lambdas": lambdas,
            "borne_inférieure": LB,
            "borne_supérieure": UB,
            "meilleur_tour": meilleur_tour,
            "iterations": k,
            "ecart_final": (UB - LB) / UB * 100 if UB > 0 else 0
        }
    
    def main():
        print("=== Résolution du TSP par la méthode du sous-gradient ===\n")
        dist = saisir_matrice_distances()
        print(f"\nMatrice des distances {dist.shape[0]}x{dist.shape[0]} saisie avec succès.")
        print("Démarrage de l'algorithme...\n")
        
        resultats = sous_gradient_tsp(dist)
        
        print("\n" + "="*70)
        print("RÉSULTATS FINAUX")
        print("="*70)
        print(f"Borne inférieure finale    : {resultats['borne_inférieure']:.2f}")
        print(f"Borne supérieure finale    : {resultats['borne_supérieure']:.2f}")
        print(f"Écart relatif final        : {resultats['ecart_final']:.2f}%")
        print(f"Nombre d'itérations        : {resultats['iterations']}")
        print(f"Longueur du meilleur tour  : {resultats['borne_supérieure']:.2f}")
        print("="*70)

    main()