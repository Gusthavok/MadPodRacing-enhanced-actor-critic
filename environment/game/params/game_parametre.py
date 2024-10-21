from numpy.random import randint
from math import sin, cos, pi

# collisions
nombre_de_dt = 10
collision_tab = [
    [True, True, True, True], 
    [True, True, True, True], 
    [True, True, True, True], 
    [True, True, True, True]
]
def collision_activee(i,j):
    return collision_tab[i][j]

# checkpoints
taille_cp = 600

# fin du jeu (en commentaire les parametres pour une vrai partie)
nombre_de_tick_max = 300 # 1000
nombre_de_tick_max_sans_cp = 100 # 100
defaite_j2_possible = False # True
defaite_j1_possible = False # True

# teleportation des pods de defense
placement_entrainement_defense_tp_x = [0 for _ in range(8)] # au plus 8 cp par carte
placement_entrainement_defense_tp_y = [0 for _ in range(8)]
orientation_defense_tp = [0 for _ in range(8)]
cp_tp = 0
def set_placement_entrainement_defense(rayon):
    global placement_entrainement_defense_tp_x, placement_entrainement_defense_tp_y, orientation_defense_tp
    for i in range(8):
        r = randint(rayon//2, rayon)
        ang = randint(-180, 180)
        placement_entrainement_defense_tp_x[i] = r * cos(ang/180*pi)
        placement_entrainement_defense_tp_y[i] = r * sin(ang/180*pi)
        orientation_defense_tp[i] = randint(-179, 181)



