# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

# defenseur :
BATCH_SIZE = 2048

# Fonction Q = Score(t) + Q(t-1)
GAMMA_START = 0.95
GAMMA_END = 0.95
GAMMA_OFFSET = 20000
GAMMA_TEMPS = 120000 # Au bout de n+GAMMA_OFFSET événements, on serra a GAMMA_START + (GAMMA_END - GAMMA_START)*(n/GAMMA_TEMPS)

# Fréquence des actions aléatoires
EPS_START = 0.5
EPS_END = 0.05
EPS_OFFSET = 10000
EPS_DECAY = 40000

# Taux pour la soft actualisation (stabilise le modèle)
TAU = 0.002

# Learning rates
LR_ACTOR = 1e-5
LR_CRITIC = 1e-5

