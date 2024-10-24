import environment.game.main as game
import environment.action_space as action_space
import environment.ressources.maps as maps
from .constante import *

def reset(choose_map = -1):
    global jeu, carte_cp, score_adv
    # Charge une nouvelle partie aléatoire
    # choisir une carte aleatoirement :
    
    if choose_map < 0:
        carte_cp, score_adv = maps.get_random_map()
    else: 
        carte_cp = maps.echantillon_carte[choose_map]
        score_adv = 1

    jeu = game.Etat_de_jeu(carte_cp, nb_tour=100)
    observation_hero, observation_adversaire = jeu.get_observation(indice=0), jeu.get_observation(indice=2)
    
    return observation_hero, observation_adversaire, False

def transform_output_1_pod(x, y, pow, shield, boost):
    puissance = max(0, min(100, 1.2*Facteur_compression_puissance*pow -10))
    
    if shield >.95:
        puissance = "SHIELD"
    elif boost>.98:
        puissance = "BOOST"
    
    return Facteur_compression_distances*x, Facteur_compression_distances*y, puissance

def transform_output(action):
    j1_pa_x, j1_pa_y, j1_pa_pow, j1_pa_shield, j1_pa_boost, j1_pd_x, j1_pd_y, j1_pd_pow, j1_pd_shield, j1_pd_boost  = action
    j1_pa_x, j1_pa_y, j1_pa_pow = transform_output_1_pod(j1_pa_x, j1_pa_y, j1_pa_pow, j1_pa_shield, j1_pa_boost)
    j1_pd_x, j1_pd_y, j1_pd_pow = transform_output_1_pod(j1_pd_x, j1_pd_y, j1_pd_pow, j1_pd_shield, j1_pd_boost)
    return j1_pa_x, j1_pa_y, j1_pa_pow, j1_pd_x, j1_pd_y, j1_pd_pow
    

def step(action_hero, action_adversaire):
    global jeu
    
    observation_hero, observation_adversaire, terminated, infos_supplementaires = jeu.etape_de_jeu(transform_output(action_hero), transform_output(action_adversaire))

    return observation_hero, observation_adversaire, terminated, infos_supplementaires

def get_cp():
    global carte_cp, score_adv
    n_cp = len(carte_cp)
    n_cp_hero = jeu.get_n_cp(joueur=0)
    n_cp_adversaire = jeu.get_n_cp(joueur=2)
    print(n_cp_hero, n_cp_adversaire)
    return n_cp_hero/score_adv, n_cp_adversaire/score_adv


def afficher():
    global jeu

    jeu.afficher()