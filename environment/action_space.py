from numpy.random import rand
from math import sqrt
from .constante import *

n = 8

def distance(couple1, couple2):
    return sqrt((couple1[0]-couple2[0])**2 + (couple1[1]-couple2[1])**2)

def sample_action(environnement):
    environnement = environnement[0, :]
    pod1a = (environnement[20+0], environnement[20+1])
    pod1d = (environnement[20+4+0], environnement[20+4+1])
    pod2a = (environnement[20+8+0], environnement[20+8+1])
    pod2d = (environnement[20+12+0], environnement[20+12+1])
    
    next_cp_1 = (environnement[0], environnement[1])
    
    next_next_cp_adv_2 = (environnement[10+2+0], environnement[10+2+1])
    
    # SHIELDS
    shield_a = rand()*0.95
    shield_d = rand()*0.95
    if distance(pod1a, pod2a)<6:
        shield_a = .5+rand()/2 # 1 chance sur 10 de shield
    if distance(pod1a, pod2d)<6:
        shield_a = .75+rand()/4 # 1 chance sur 5 de shield
    if distance(pod1d, pod2a)<6:
        shield_d = .75+rand()/4
    if distance(pod1d, pod2d)<6:
        shield_d = .5+rand()/2
    
    # BOOSTS : 
    boost_a = rand() # 1/50
    boost_d = rand()*.99 # boost  1/100
    
    # Direction puissance a
    ecart_atq = 5000
    dir_x_a = next_cp_1[0]+(2*ecart_atq*rand()-ecart_atq)
    dir_y_a = next_cp_1[1]+(2*ecart_atq*rand()-ecart_atq)
    pow_a = 100*rand()#100*(1-rand()**3)
    

    # Direction puissance d
    
    eacrt_def = 3000
    if rand()<.5:
        dir_x_d = pod2a[0]+(2*eacrt_def*rand()-eacrt_def)
        dir_y_d = pod2a[1]+(2*eacrt_def*rand()-eacrt_def)
        pow_d = 100*(1-rand()**3)
    else:
        dir_x_d = next_next_cp_adv_2[0]+(2*eacrt_def*rand()-eacrt_def)
        dir_y_d = next_next_cp_adv_2[1]+(2*eacrt_def*rand()-eacrt_def)
        pow_d = 100*(1-rand()**3)
    
    # pour 1 pod : x, y, pow, shield, boost
    
    if rand()<.5:
        return (float(dir_x_a/Facteur_compression_distances), 
                float(dir_y_a/Facteur_compression_distances), 
                pow_a/Facteur_compression_puissance, 
                shield_a, 
                boost_a,
                float(dir_x_d/Facteur_compression_distances), 
                float(dir_y_d/Facteur_compression_distances), 
                pow_d/Facteur_compression_puissance, 
                shield_d, 
                boost_d,)
    else:
        return ((18000*rand()-2000)/Facteur_compression_distances, 
                (12000*rand()-2000)/Facteur_compression_distances, 
                (100*rand())/Facteur_compression_puissance, 
                6*rand()**3 - 2, 
                6*rand()**3 - 2,
                (18000*rand()-2000)/Facteur_compression_distances, 
                (12000*rand()-2000)/Facteur_compression_distances, 
                (100*rand())/Facteur_compression_puissance, 
                6*rand()**3 - 2, 
                6*rand()**3 - 2)
