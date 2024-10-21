import environment.game.params.game_parametre as parametre
from environment.game.calcul_angle import *
from math import pi, cos, sin, sqrt
from numpy.random import randint

# la fonction analyse() renvoie une valeur adéquate de la puissance demandée par le bot
def analyse(val, bit_boost, valshield):
    if val == "SHIELD":
        return 0, bit_boost, 3
    elif valshield>0: # S'il ne reshield pas, on se fiche de son action
        return 0, bit_boost, valshield-1
    elif isinstance(val, int):
        return max(0, min(100, val)), bit_boost, 0
    elif val == "BOOST":
        if bit_boost:
            return 650, False, 0
        else:
            return 100, False, 0
    else:
        return 0, bit_boost, 0

# la fonction deplacement() modifie en place la case 6 = orientation de chaque pod
# la fonction deplacement() modifie en place les cases 4 et 5 = vitesse de chaque pod
# la fonction deplacement() modifie en place les cases 2 et 3 = position de chaque pod
def deplacement(lpod, premierTour, boost_1, boost_2, entrainement_atq):
    orientation = []
    nv_vx = []
    nv_vy = []
    nv_x = []
    nv_y = []

    for indice, (pod, decision) in enumerate(lpod):
        if indice<=1:
            puissance, boost_1, pod[8] = analyse(decision[2], boost_1, pod[8])
        else:
            puissance, boost_2, pod[8]= analyse(decision[2], boost_2, pod[8])
        # Orientation :
        if premierTour:
            orientation.append(int(angle(decision[0] - pod[2], decision[1] - pod[3])))
        else:
            wanted_orientation = angle(decision[0] - pod[2], decision[1] - pod[3])
            ecart = normalized(wanted_orientation - pod[6])
            if abs(ecart) <= 18:
                orientation.append(int(wanted_orientation))
            else:
                if ecart > 0:
                    orientation.append(normalized(pod[6] + 18))
                else:
                    orientation.append(normalized(pod[6] - 18))
         
        nv_vx.append(puissance*cos(orientation[-1]*pi/180)+pod[4]) # nouvelle vitesse 
        nv_vy.append(puissance*sin(orientation[-1]*pi/180)+pod[5]) # nouvelle vitesse 
        nv_x.append(pod[2])
        nv_y.append(pod[3])



    proche= True

    if not proche:
        dt = 1
        ndt = 1
    else:
        dt = 1/parametre.nombre_de_dt
        ndt = parametre.nombre_de_dt

    rebond_frat = False
    rebond_enem = False
    for _ in range(ndt):
        for i in range(4):
            nv_x[i] += nv_vx[i] * dt
            nv_y[i] += nv_vy[i] * dt
            for j in range(i+1,4):

                if parametre.collision_activee(i,j) and (nv_x[i]-nv_x[j])**2 + (nv_y[i]-nv_y[j])**2 < 800**2: # 800 est le diamètre d'un pod
                    
                    if i==0 and j==1:
                        rebond_frat = True
                    if i==1 and j==2:
                        rebond_enem = True
                    
                    angle_collision = angle(nv_x[j]-nv_x[i], nv_y[j]-nv_y[i])

                    angle_vi = angle(nv_vx[i], nv_vy[i]) - angle_collision
                    angle_vj = angle(nv_vx[j], nv_vy[j]) - angle_collision

                    v_orth_i = sin(angle_vi*pi/180) * sqrt(nv_vx[i]**2 + nv_vy[i]**2)
                    v_para_i = cos(angle_vi*pi/180) * sqrt(nv_vx[i]**2 + nv_vy[i]**2)

                    v_orth_j = sin(angle_vj*pi/180) * sqrt(nv_vx[j]**2 + nv_vy[j]**2)
                    v_para_j = cos(angle_vj*pi/180) * sqrt(nv_vx[j]**2 + nv_vy[j]**2)

                    if v_para_j - v_para_i<0:
                        if lpod[i][0][8] == 3:
                            mi = 10
                        else:
                            mi = 1

                        if lpod[j][0][8] == 3:
                            mj = 10
                        else:
                            mj = 1
                        
                        nv_para_i = (mi-mj)/(mi+mj) * v_para_i + (2*mj)/(mi+mj) * v_para_j
                        nv_para_j = (mj-mi)/(mi+mj) * v_para_j + (2*mi)/(mi+mj) * v_para_i

                        #CHANGEMENT DES VITESSES ORTHOGONALES, A CAUSE DE LA REGLE BIZARRE QUI IMPOSE 120 Minimum
                        total_impulsion = abs(nv_para_i) + abs(nv_para_j)
                        if total_impulsion < 120:
                            nv_para_i = 120 * nv_para_i/(total_impulsion + 0.000001)
                            nv_para_j = 120 * nv_para_j/(total_impulsion + 0.000001)



                        nv_angle_vi_rep = angle(nv_para_i, v_orth_i) # angle dans le repère parallèle, orthogonal à la collision
                        nv_angle_vj_rep = angle(nv_para_j, v_orth_j)

                        nv_x[i] -= nv_vx[i] * dt
                        nv_y[i] -= nv_vy[i] * dt

                        nv_vx[i] = cos((nv_angle_vi_rep+angle_collision)*pi/180) * sqrt(v_orth_i**2 + nv_para_i**2)
                        nv_vy[i] = sin((nv_angle_vi_rep+angle_collision)*pi/180) * sqrt(v_orth_i**2 + nv_para_i**2)

                        nv_vx[j] = cos((nv_angle_vj_rep+angle_collision)*pi/180) * sqrt(v_orth_j**2 + nv_para_j**2)
                        nv_vy[j] = sin((nv_angle_vj_rep+angle_collision)*pi/180) * sqrt(v_orth_j**2 + nv_para_j**2)

                        nv_x[i] += nv_vx[i] * dt
                        nv_y[i] += nv_vy[i] * dt

    
    for indice, (pod, decision) in enumerate(lpod):
        pod[6] = orientation[indice]

        pod[4] = 0.85*(nv_vx[indice])
        pod[5] = 0.85*(nv_vy[indice])

        pod[2] = nv_x[indice]
        pod[3] = nv_y[indice]

        # Modifications en place
    return boost_1, boost_2, rebond_frat, rebond_enem

# Valide le passage d'un cp et actualise le prochain cp d'un pod
def cp_valide(lpod, carte_cp, cp_avant_tp, entrainement_atq):
    # modifie en place les cases 0 (nb de tour) 1 (prochain cp) et 7 (nb de tour sans passer de cp) de CHAQUE pod
    for i, (pod, dec) in enumerate(lpod):
        xcp, ycp = carte_cp[pod[1]]
        if (pod[2]-xcp)**2 + (pod[3]-ycp)**2 < parametre.taille_cp**2:
            pod[1]+=1
            if pod[1] >= len(carte_cp):
                pod[0]+=1
                pod[1]=0
            pod[7] = 0

            # téléportation pour l'entrainement avec défense 
            if entrainement_atq==1:
                if i==0 and (pod[0]*len(carte_cp) + pod[1])%cp_avant_tp == cp_avant_tp - 1:
                    cp_tp = (pod[1]+parametre.cp_tp)%len(carte_cp)
                    lpod[3][0][2], lpod[3][0][3] = carte_cp[cp_tp][0] + parametre.placement_entrainement_defense_tp_x[cp_tp], carte_cp[cp_tp][1] + parametre.placement_entrainement_defense_tp_y[cp_tp]
                    lpod[3][0][4], lpod[3][0][5] = 0, 0
                    lpod[3][0][6] = parametre.orientation_defense_tp[cp_tp]
            elif entrainement_atq==-1:
                if i==2 and (pod[0]*len(carte_cp) + pod[1])%cp_avant_tp == cp_avant_tp - 1:
                    cp_tp = (pod[1]+parametre.cp_tp)%len(carte_cp)
                    lpod[1][0][2], lpod[1][0][3] = carte_cp[cp_tp][0] + parametre.placement_entrainement_defense_tp_x[cp_tp], carte_cp[cp_tp][1] + parametre.placement_entrainement_defense_tp_y[cp_tp]
                    lpod[1][0][4], lpod[1][0][5] = 0, 0
                    lpod[1][0][6] = parametre.orientation_defense_tp[cp_tp]

        else:
            pod[7]+=1

# Dans le cas ou plusieurs courses sont lancées parallélement, cette fonction permet de savoir si toutes les courses sont terminées. 
def fin(pods):
    bit_fin, _, _, _, _, _, _ = pods
    return bit_fin != 0

#initialisation de la position des pods initial probablement à modifier.
def pods_start(carte):
    # liste de 7-uplet :
    # un bit pour savoir si la partie est finie(0) gagnée par J1 (1) ou gagnee par J2 (2)
    # suivie de 4 pods : pod défini par une liste de éléments : [numeroTour(commence à 0), numeroProchainCP(de 0 à nb_cp-1), x, y, vx, vy, orientation, nombre de temps depuis le dernier Cp traversé, Shield Couldown]
    # un bit pour savoir si le J1 a utilisé le Boost
    # un bit pour savoir si le J2 a utilisé le Boost
    
    # Pour la position de départ des pods, on les place sur la même droite, (la perpendiculaire à l'axe CP_0 - CP_1 passant par le CP_0) 

    ang = angle(carte[1][0]-carte[0][0], carte[1][1]-carte[0][1])
    ux = -sin(ang*pi/180)
    uy = cos(ang*pi/180)
    orientation = randint(-180, 180)


    return (0, 
        [0,1, carte[0][0] + ux*500, carte[0][1] + uy*500, 0, 0, orientation, 0, 0],
        [0,1, carte[0][0] - ux*500, carte[0][1] - uy*500, 0, 0, orientation, 0, 0],
        [0,1, carte[0][0] + ux*1500, carte[0][1] + uy*1500, 0, 0, orientation, 0, 0],
        [0,1, carte[0][0] - ux*1500, carte[0][1] - uy*1500, 0, 0, orientation, 0, 0],
        True, 
        True)
