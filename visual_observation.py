import matplotlib.pyplot as plt

import torch
import torch.optim as optim

import environment as env

from training_utils.joueur import Joueur, Model
from training_utils.actor_critic import ReplayMemory, select_action, optimize, optimize_actor, soft_update
from training_utils.graph_train import plot_durations
from training_utils.learning_parameters import *

from models.actor.version1.architecture import Actor_v1
from models.critic.version1.architecture import Critic_v1


from models.actor.version2.architecture import Actor_v2
from models.critic.version2.architecture import Critic_v2

from models.actor.version3.architecture import Actor_v3
from models.critic.version3.architecture import Critic_v3

from models.actor.version4.architecture import Actor_v4
from models.critic.version4.architecture import Critic_v4



from environment.action_space import sample_action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = env.action_space.n
n_observations = 40
proba_moov_random = .0

hero_actor_classe = Actor_v2
hero_actor_model_version = 'version2'
hero_actor_save_name = 'none'
hero_actor_reload_name = 'gen7_bis'

hero_critic_classe = Critic_v2
hero_critic_model_version = 'version2'
hero_critic_save_name = 'none'
hero_critic_reload_name = 'gen1'


is_adversaire = True

adv_actor_classe = Actor_v2
adv_actor_model_version = 'version2'
adv_actor_save_name = 'none'
adv_actor_reload_name = 'gen2'

adv_critic_classe = Critic_v2
adv_critic_model_version = 'version2'
adv_critic_save_name = 'none'
adv_critic_reload_name = 'gen0'


hero = Joueur(
    Model(hero_actor_model_version, hero_actor_classe, hero_actor_save_name, hero_actor_reload_name, n_observations, n_actions, 'actor', device), 
    Model(hero_critic_model_version, hero_critic_classe, hero_critic_save_name, hero_critic_reload_name, n_observations, n_actions, 'critic', device),     
    Model(hero_critic_model_version, hero_critic_classe, hero_critic_save_name, hero_critic_reload_name, n_observations, n_actions, 'critic', device),     
    eval_mode = True  
)

adv = Joueur(
    Model(adv_actor_model_version, adv_actor_classe, adv_actor_save_name, adv_actor_reload_name, n_observations, n_actions, 'actor', device), 
    Model(adv_critic_model_version, adv_critic_classe, adv_critic_save_name, adv_critic_reload_name, n_observations, n_actions, 'critic', device),
    Model(adv_critic_model_version, adv_critic_classe, adv_critic_save_name, adv_critic_reload_name, n_observations, n_actions, 'critic', device),
    eval_mode = True  
)


steps_done = 0
score_hero = []
score_adv = []
num_episodes = 600


for i_episode in range(num_episodes):

    # Initialize the environment and get its state
    observation_hero, observation_adversaire, info = env.reset()

    state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
    state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

    t=-1
    while True :
        t+=1

        action_hero = select_action(state_hero, hero.actor.net, sample_action, device, proba_moov_random)
        if is_adversaire:
            action_adv = adv.action(state_adversaire)
        else:
            action_adv = torch.tensor(sample_action())


        observation_hero, observation_adversaire, reward, terminated, _ = env.step(action_hero, action_adv)
        
        state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
        state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

        
        if terminated:
            env.afficher()
            break

