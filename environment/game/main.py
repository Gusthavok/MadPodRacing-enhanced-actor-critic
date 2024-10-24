import environment.game.params.game_parametre as parametre
from environment.game.game_main_functions import pods_start, deplacement, cp_valide, fin
from environment.game.affichage_graphique import affgame
from environment.game.calcul_angle import normalized, norme_angle, next_item, angle, norme
from math import pi, cos, sin, sqrt, exp
from ..constante import *

def norm_ang(a):
    return normalized(a)/18
def norm_dist(d):
    return d/5000
def norm_vit(v):
    return v/500


class Etat_de_jeu:
    def __init__(self, carte_cp, nb_tour, cp_avant_teleportation = 100, entrainement_attaque = 1):
        self.pods = pods_start(carte_cp)

        self.carte_cp = carte_cp
        self.nb_tour = nb_tour
        self.cp_avant_teleportation = cp_avant_teleportation
        self.entrainement_attaque = entrainement_attaque

        self.memoire = [[self.pods[1].copy(), self.pods[2].copy(), self.pods[3].copy(), self.pods[4].copy()]]

        self.premier_tour = True

        self.tick = 0

    def etape_de_jeu(self, action_j1, action_j2):

        j1_pa_x, j1_pa_y, j1_pa_pow, j1_pd_x, j1_pd_y, j1_pd_pow = action_j1
        j2_pa_x, j2_pa_y, j2_pa_pow, j2_pd_x, j2_pd_y, j2_pd_pow = action_j2

        self.tick+=1
        bit_fin, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2 = self.pods
        if bit_fin == 0:

            reponse_j1_a, reponse_j1_b = (j1_pa_x, j1_pa_y, j1_pa_pow), (j1_pd_x, j1_pd_y, j1_pd_pow)
            reponse_j2_a, reponse_j2_b = (j2_pa_x, j2_pa_y, j2_pa_pow), (j2_pd_x, j2_pd_y, j2_pd_pow)

            lpod = [(pod_j1_a, reponse_j1_a), (pod_j1_b, reponse_j1_b), (pod_j2_a, reponse_j2_a), (pod_j2_b, reponse_j2_b)]
            
    
            
            # la fonction deplacement() modifie en place la case 6 = orientation de chaque pod
            # la fonction deplacement() modifie en place les cases 4 et 5 = vitesse de chaque pod
            # la fonction deplacement() modifie en place les cases 2 et 3 = position de chaque pod

            boost_j1, boost_j2, rebond_fratricide, rebond_ennemi = deplacement(lpod, self.premier_tour, boost_j1, boost_j2, self.entrainement_attaque) 


            cp_valide(lpod, self.carte_cp, self.cp_avant_teleportation, self.entrainement_attaque) # modifie en place les cases 0 (nb de tour) 1 (prochain cp) et 7 (nb de tour sans passer de cp) de chaque pod

            # Change le bit de fin de manière 
            if parametre.defaite_j2_possible and (pod_j2_a[7]>parametre.nombre_de_tick_max_sans_cp and pod_j2_b[7]>parametre.nombre_de_tick_max_sans_cp) or (pod_j1_a[0] == self.nb_tour and pod_j1_a[1] == 1) or (pod_j1_b[0] == self.nb_tour and pod_j1_b[1] == 1): # (pod_j1_a[0] == nb_tour and pod_j1_a[1] == 1) car pour finir la carte, il faut revenir jusqu'au point de départ (CP dindice 0)
                self.pods = (1, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2)
            elif parametre.defaite_j1_possible and (pod_j1_a[7]>parametre.nombre_de_tick_max_sans_cp and pod_j1_b[7]>parametre.nombre_de_tick_max_sans_cp) or (pod_j2_a[0] == self.nb_tour and pod_j2_a[1] == 1) or (pod_j2_b[0] == self.nb_tour and pod_j2_b[1] == 1):
                self.pods = (2, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2)
            else:
                self.pods = (0, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2)
            
            l = [self.pods[1].copy(), self.pods[2].copy(), self.pods[3].copy(), self.pods[4].copy()]
            self.memoire.append(l)

        self.premier_tour = False

        infos_supplementaires = {"impact": rebond_fratricide or rebond_ennemi}
        return self.get_observation(indice = 0), self.get_observation(indice = 2), (fin(self.pods) or self.tick>parametre.nombre_de_tick_max), infos_supplementaires
        # version 1 : 
        # return (self.get_observation(indice = 0), self.get_observation(indice = 1)), (self.get_observation(indice = 2), self.get_observation(indice = 3)), self.get_reward_atq(rebond_fratricide), self.get_reward_dfs(rebond_fratricide, rebond_ennemi), (fin(self.pods) or self.tick>parametre.nombre_de_tick_max)

    def get_n_cp(self, joueur = 0):
        return len(self.carte_cp)* self.memoire[-1][joueur][0] + self.memoire[-1][joueur][1] -1

    def get_distance_cp(self, indice_cp1, indice_cp2):
        return sqrt((self.carte_cp[indice_cp2][0] - self.carte_cp[indice_cp1][0])**2 + (self.carte_cp[indice_cp2][0] - self.carte_cp[indice_cp1][0])**2)
# A MODIF
    def get_observation(self, indice):
        if indice ==0:
            order_pods = [0, 1, 2, 3]
        elif indice ==2:
            order_pods = [2, 3, 0, 1]
        else:
            raise ValueError("l'indice de la fonction get_observation ne correspond à rien")
                
        l_objet_physiques = []
        l_objet_non_physiques = []
        l_autre = []

        for ind_obs in range(2, 6):
            for ind_pod in order_pods:
                l_objet_physiques.append(self.memoire[-1][ind_pod][ind_obs]/Facteur_compression_distances)


    
        for ind_pod in [order_pods[0], order_pods[2]]:
            indice_cp = self.memoire[-1][ind_pod][1]
            for depl in range(5):
                x, y = next_item(self.carte_cp, indice_cp, depl)
                l_objet_non_physiques.append(x/Facteur_compression_distances)
                l_objet_non_physiques.append(y/Facteur_compression_distances)


        for ind_pod in order_pods:
            ang = self.memoire[-1][ind_pod][6]/180 * pi
            l_objet_physiques.append(cos(ang)/Facteur_compression_cos_sin)
            l_objet_physiques.append(sin(ang)/Facteur_compression_cos_sin)

        bit_fin, pod_j1_a, pod_j1_b, pod_j2_a, pod_j2_b, boost_j1, boost_j2 = self.pods

        for ind_pod in order_pods:
            if ind_pod == 0:
                l_autre.append(float(boost_j1))
                l_autre.append(len(self.carte_cp)*(self.nb_tour-self.memoire[-1][ind_pod][0]) + self.memoire[-1][ind_pod][1])
                l_autre.append((self.memoire[-1][ind_pod][7])/20)
            if ind_pod == 2:
                l_autre.append(float(boost_j2))
                l_autre.append(len(self.carte_cp)*(self.nb_tour-self.memoire[-1][ind_pod][0]) + self.memoire[-1][ind_pod][1])
                l_autre.append((self.memoire[-1][ind_pod][7])/20)
    
        for ind_pod in order_pods:
            l_autre.append(self.memoire[-1][ind_pod][8])
        
        ecart_adverse = self.ecart_pods(order_pods[0], order_pods[2])
        return l_objet_non_physiques + l_objet_physiques + l_autre  + [ecart_adverse] # l_objet_non_physiques : 20, l_objet_physiques : 24,  l_autre :  10, ecart_adv:1

    def ecart_pods(self, indice_pod_1, indice_pod_2):
        nombre_cp=len(self.carte_cp)
        
        indice_cp_1 = nombre_cp*self.memoire[-1][indice_pod_1][0] +self.memoire[-1][indice_pod_1][1]
        indice_cp_2 = nombre_cp*self.memoire[-1][indice_pod_2][0] +self.memoire[-1][indice_pod_2][1]
        
        if indice_cp_1>= indice_cp_2:
            
            distance = 0
            for k in range(indice_cp_2, indice_cp_1):
                distance += self.get_distance_cp(k%nombre_cp, (k+1)%nombre_cp)
            # distance du pod 2 à son cp
            distance += sqrt((self.memoire[-1][indice_pod_2][2] - self.carte_cp[indice_cp_2][0])**2 + (self.memoire[-1][indice_pod_2][3] - self.carte_cp[indice_cp_2][1])**2)
            # distance du pod 1 à son cp
            distance -= sqrt((self.memoire[-1][indice_pod_1][2] - self.carte_cp[indice_cp_1][0])**2 + (self.memoire[-1][indice_pod_1][3] - self.carte_cp[indice_cp_1][1])**2)
            return distance/Facteur_compression_distances
        else:
            return - self.ecart_pods(indice_pod_2, indice_pod_1)

    def afficher(self):
        exemple = self.memoire
        lpod1 = []
        for _, l in enumerate(exemple):
            lpod1.append([[int(l[i][2]), int(l[i][3]), l[i][6]/180*pi]for i in range(4)])
        carte_cp_reduite = []

        for x,y in self.carte_cp:
            carte_cp_reduite.append((x, y))
        affgame(carte_cp_reduite, lpod1)
        
