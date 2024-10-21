from numpy.random import rand
from .constante import *

n = 8

def sample_action():
    # return rand((8))
    return ((18000*rand()-2000)/Facteur_compression_distances, 
            (12000*rand()-2000)/Facteur_compression_distances, 
            (100*rand())/Facteur_compression_puissance, 
            rand(), 
            rand(),
            (18000*rand()-2000)/Facteur_compression_distances, 
            (12000*rand()-2000)/Facteur_compression_distances, 
            (100*rand())/Facteur_compression_puissance, 
            rand(), 
            rand(),)
