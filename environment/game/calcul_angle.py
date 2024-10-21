from math import atan, pi, sqrt
# la fonction angle() renvoie l'angle en degrès du vecteur de coordonnées vectx, vecty sachant que l'EST corespond à 0° et le SUD à 90°
def angle(vectx,vecty):
    if vectx == 0:
        if vecty>0:
            return 90
        else: #comprend le cas du vecteur (0,0)
            return -90
    elif vectx >0:
        return atan(vecty/vectx)*180/pi
    else:
        return normalized(180+atan(vecty/vectx)*180/pi)

# normalized(ang) renvoie le même angle ang à 360 degrès prés, dans [-180, 180[
def normalized(ang):
    return (int(ang+180)%360 - 180) # RMQ peut renvoyer -180 au lieu de +180, pas grave. 

def norme(x, y):
    return sqrt(x**2 + y**2)

def norme_angle(z1, z2, referentiel_angle):
    x1, y1 = z1
    x2, y2 = z2
    return norme(x2-x1, y2-y1), normalized(angle(x2-x1, y2-y1) - referentiel_angle)

def next_item(liste, indice, augmentation):
    return liste[(indice+augmentation)%len(liste)]