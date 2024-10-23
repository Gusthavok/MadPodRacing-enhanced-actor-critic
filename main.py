import matplotlib.pyplot as plt
from numpy import exp

import torch
import torch.optim as optim

import environment as env

from training_utils.joueur import Joueur, Model
from training_utils.actor_critic import ReplayMemory, select_action, optimize_predictor, optimize_critic, optimize_actor, soft_update
from training_utils.graph_train import plot_on_4_diagrams
from training_utils.learning_parameters import *
from models.actor.version1.architecture import Actor
from models.critic.version1.architecture import Critic
from models.predictor.version1.architecture import Predictor
from environment.action_space import sample_action

def gamma(steps):
    # Initialement, on avait : 
    # GAMMA_START sur [0, GAMMA_OFFSET]
    # GAMMA_END sur [GAMMA_OFFSET + GAMMA_TEMPS, +inf[
    # Pente affine reliant continuement les deux morceaux sur l'interval manquant
    # Toutefois, comme la somme des q^n vaut (1-q)^-1, il est plus cohérents que q (ici GAMMA) suive une loi de la forme 1-(variation_linéaire^-1)
    

    def fonction_vers_somme(x):
        return 1/(1-x)
    

    if steps<GAMMA_OFFSET:
        return GAMMA_START
    elif steps>GAMMA_OFFSET+GAMMA_TEMPS:
        return GAMMA_END
    else:
        p_start = fonction_vers_somme(GAMMA_START)
        p_end = fonction_vers_somme(GAMMA_END)
        return 1 - 1/(p_start + (p_end-p_start) * (steps-GAMMA_OFFSET)/GAMMA_TEMPS)

def epsilon(steps):
    if steps<EPS_OFFSET:
        return EPS_START
    else:
        return EPS_END + (EPS_START - EPS_END) * exp(-1. * (steps-EPS_OFFSET) / EPS_DECAY)

def plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, gamma_value, epsilon_value, show_results=False):
    dict11 = {
        "titre": "checkpoints",
        "with_mean": True,
        "hero": score_hero,
        "adversaire": score_adv
    }
    dict12 = {
        "titre": "training parameters",
        "gamma": gamma_value,
        "epsilon": epsilon_value
    }
    dict21 = {
        "titre": "-q loss: actor (NORAMLIZED ON GAMMA)",
        "loss": l_loss_actor
    }
    dict22 = {
        "titre": "MSE loss: critic (NORMALIZED ON Q VALUE)",
        "loss": l_loss_critic
    }

    plot_on_4_diagrams(dict11, dict12, dict21, dict22)

def hard_critic(env):
    return 0

def main():
    plt.ion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hero_actor_classe = Actor
    hero_actor_model_version = 'version1'
    hero_actor_save_name = 'gen_1'
    hero_actor_reload_name = ''

    hero_critic_classe = Critic
    hero_critic_model_version = 'version1'
    hero_critic_save_name = 'gen_1'
    hero_critic_reload_name = ''
    
    actor = Model(hero_actor_model_version, hero_actor_classe, hero_actor_save_name, hero_actor_reload_name, 'actor', device)
    critic = Model(hero_critic_model_version, hero_critic_classe, hero_critic_save_name, hero_critic_reload_name, 'critic', device) 
    critic_smooth = Model(hero_critic_model_version, hero_critic_classe, hero_critic_save_name, hero_critic_reload_name, 'critic', device)   

    predictor_net = Predictor()
    predictor_net.load_state_dict(torch.load('./models/Predictor/version1/safetensor/test', map_location=device))

    hero = Joueur(actor, critic, critic_smooth, eval_mode = False)

    adv = Joueur(actor, critic, critic_smooth, eval_mode = False)

    optimizer_actor = optim.AdamW(hero.actor.net.parameters(), lr=LR_ACTOR, amsgrad=True)
    optimizer_critic = optim.AdamW(hero.critic.net.parameters(), lr=LR_CRITIC, amsgrad=True)
    optimizer_predictor = optim.AdamW(predictor_net.parameters(), lr=1e-4, amsgrad=True)
    memory = ReplayMemory(24000)


    steps_done = 0
    score_hero = []
    score_adv = []
    l_loss_actor = []
    l_loss_critic = []

    l_gamma, l_epsilon = [], []
    num_episodes = 600

    for i_episode in range(num_episodes):

        observation_hero, observation_adversaire, _ = env.reset()

        # print(observation_hero)
        state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
        state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

        t=-1
        loss_actor, loss_critic = 0, 0
        while True :
            t+=1

            action_hero = select_action(state_hero, hero.actor.net, sample_action, device, epsilon(steps_done))
            action_adv = select_action(state_adversaire, adv.actor.net, sample_action, device, epsilon(steps_done))
            steps_done += 1


            observation_hero, observation_adversaire, terminated, info_supplementaires = env.step(action_hero, action_adv)
            action_hero = action_hero.unsqueeze(0)
            action_adv = action_adv.unsqueeze(0)
            

            next_state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
            next_state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state_hero, action_hero, action_adv, next_state_hero, next_state_hero[:, -1])
            memory.push(state_adversaire, action_adv, action_hero, next_state_adversaire, next_state_adversaire[:, -1])

            state_hero = next_state_hero
            state_adversaire = next_state_adversaire

            l_a = optimize_actor(memory, hero.actor.net, hero.critic.net, optimizer_actor, predictor_net, BATCH_SIZE)
            l_c = optimize_critic(memory, hero.critic.net, hero.critic_smooth.net, optimizer_critic, gamma(steps_done), BATCH_SIZE)
            soft_update(hero.critic.net, hero.critic_smooth.net)
            
            loss_actor+=l_a
            loss_critic+=l_c

            if terminated:
                
                for _ in range(5):
                    optimize_predictor(memory, predictor_net, optimizer_predictor, BATCH_SIZE=512)
                loss_actor = float(torch.tensor(loss_actor).to('cpu'))
                loss_critic = float(torch.tensor(loss_critic).to('cpu'))

                if loss_actor>0:
                    norm_loss_critic=0.2
                else:
                    norm_loss_critic = min(.2, loss_critic/(1e-4-loss_actor))

                if norm_loss_critic>=.2:
                    norm_loss_actor=0
                else:
                    norm_loss_actor=(loss_actor/(t+1))*(1-gamma(steps_done))
                l_loss_actor.append(norm_loss_actor) # Normalisée sur le fait que la loss de l'actor donc la q value va croitre avec la valeur de gamma ((1-gamma)^-1)
                l_loss_critic.append(norm_loss_critic) # on normalise par rapport à la valeur des score qu'il modélise. 

                l_gamma.append(gamma(steps_done))
                l_epsilon.append(epsilon(steps_done))

                n_cp_hero, n_cp_adversaire = env.get_cp()

                score_hero.append(n_cp_hero)
                score_adv.append(n_cp_adversaire)

                torch.save(hero.actor.net.state_dict(), f'./models/actor/{hero.actor.version}/safetensor/{hero.actor.save_name}')
                torch.save(hero.critic.net.state_dict(), f'./models/critic/{hero.critic.version}/safetensor/{hero.critic.save_name}')

                plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, l_gamma, l_epsilon)

                break


    print('Complete')
    plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, l_gamma, l_epsilon, show_results=True)
    plt.ioff()
    plt.show()