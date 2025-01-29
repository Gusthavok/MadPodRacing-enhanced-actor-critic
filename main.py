import matplotlib.pyplot as plt
from numpy import exp, sqrt, random

import torch
import torch.optim as optim

from collections import deque

import environment as env

from training_utils.joueur import Joueur, Model
from training_utils.actor_critic import ReplayMemory, select_action, optimize_predictor, optimize_critic, optimize_actor, soft_update, Transition
from training_utils.graph_train import plot_on_6_diagrams
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

def plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, gamma_value, epsilon_value,eqm_critic_liste, critic_value, critic_value_expected, hard_critic_value, l_loss_predictor, l_max_loss_predictor, show_results=False):
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
        "titre": "-q loss: actor",
        "loss": l_loss_actor
    }
    dict22 = {
        "titre": "critic precision",
        # "EQM": eqm_critic_liste, 
        "mean of c(s_{n+1})": critic_value, 
        "mean of c(d(s_n, a_1, a_2))": critic_value_expected,
        "hard_critic_value(s_n)": hard_critic_value
    }
    
    dict31 = {
        "titre": "Predictor precision ",
        "EQM": l_loss_predictor,
        # "max_loss": l_max_loss_predictor,
    }
    dict32 = {
        "titre": "MSE loss: critic",
        "loss": l_loss_critic
    }
    plot_on_6_diagrams(dict11, dict12, dict21, dict22, dict31, dict32)

def hard_critic(env):
    return 0

def main():
    plt.ion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hero_actor_classe = Actor
    hero_actor_model_version = 'version1'
    hero_actor_save_name = 'gen2-1-64-bis'
    hero_actor_reload_name = 'gen1-1-64-bis'

    hero_critic_classe = Critic
    hero_critic_model_version = 'version1'
    hero_critic_save_name = 'gen_3'
    hero_critic_reload_name = 'gen_2'
    
    actor = Model(hero_actor_model_version, hero_actor_classe, hero_actor_save_name, hero_actor_reload_name, 'actor', device)
    critic = Model(hero_critic_model_version, hero_critic_classe, hero_critic_save_name, hero_critic_reload_name, 'critic', device) 
    critic_smooth = Model(hero_critic_model_version, hero_critic_classe, hero_critic_save_name, hero_critic_reload_name, 'critic', device)   

    predictor_net = Predictor().to(device)
    predictor_net.load_state_dict(torch.load('./models/predictor/version1/safetensor/test_cuda', map_location=device))

    hero = Joueur(actor, critic, critic_smooth, eval_mode = False)

    is_adv=False
    adv = Joueur(actor, critic, critic_smooth, eval_mode = False)

    optimizer_actor = optim.AdamW(hero.actor.net.parameters(), lr=LR_ACTOR, amsgrad=True)
    optimizer_critic = optim.AdamW(hero.critic.net.parameters(), lr=LR_CRITIC, amsgrad=True)
    optimizer_predictor = optim.AdamW(predictor_net.parameters(), lr=1e-5, amsgrad=True)
    memory = ReplayMemory(128*1024)


    steps_done = 0
    score_hero = []
    score_adv = []
    l_loss_actor = []
    l_loss_critic = []

    l_gamma, l_epsilon = [], []
    num_episodes = 600
    
    eqm_critic_liste = []
    critic_value = []
    critic_value_expected = []
    hard_critic_value = []
    l_loss_predictor  = []
    l_max_loss_predictor = []

    num_game_before_learning_game = 4 # if =4 :  4 non learning game, then 1 learning
    learning_circles = 1

    loss_actor_dq, loss_critic_dq, loss_pred_dq= deque([0], maxlen=300), deque([0], maxlen=300), deque([0], maxlen=300)

    
    for i_episode in range(num_episodes):

        observation_hero, observation_adversaire, _ = env.reset()

        # print(observation_hero)
        state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
        state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

        t=-1
        critic_sum, critic_expected_sum, hard_critic_sum = 0, 0, 0
        
        while True :
            t+=1
            steps_done += 1

            action_hero = select_action(state_hero, hero.actor.net, sample_action, device, epsilon(steps_done))
            if is_adv:
                action_adv = select_action(state_adversaire, adv.actor.net, sample_action, device, 3*epsilon(steps_done))
            else:
                action_adv = torch.tensor(sample_action(state_adversaire)).to(device)


            observation_hero, observation_adversaire, terminated, info_supplementaires = env.step(action_hero, action_adv)
            action_hero = action_hero.unsqueeze(0)
            action_adv = action_adv.unsqueeze(0)
            
            next_state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
            next_state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state_hero, action_hero, action_adv, next_state_hero, state_hero[:, -1])
            # memory.push(state_adversaire, action_adv, action_hero, next_state_adversaire, next_state_adversaire[:, -1])

            critic_sum += hero.critic.net(next_state_hero).clone().detach()
            critic_expected_sum += hero.critic.net(torch.cat((state_hero[:, :20], predictor_net(state_hero, action_hero, action_adv)), dim=1)).clone().detach()
            hard_critic_sum += next_state_hero[0, -1]
            
            state_hero = next_state_hero
            state_adversaire = next_state_adversaire

            if learning_circles==num_game_before_learning_game:
                if len(memory) >= BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    if l_loss_predictor[-1]>1*(1-random.rand()):
                        loss_pred_dq.append(optimize_predictor(batch, predictor_net, optimizer_predictor).clone().detach().to('cpu'))
                    elif abs(critic_value[-1]-critic_value_expected[-1])>1*(1-random.rand()):
                        loss_pred_dq.append(optimize_predictor(batch, predictor_net, optimizer_predictor).clone().detach().to('cpu'))
                        loss_critic_dq.append(optimize_critic(batch, hero.critic.net, hero.critic_smooth.net, optimizer_critic, gamma(steps_done)).clone().detach().to('cpu'))
                    else:
                        loss_pred_dq.append(optimize_predictor(batch, predictor_net, optimizer_predictor).clone().detach().to('cpu'))
                        loss_critic_dq.append(optimize_critic(batch, hero.critic.net, hero.critic_smooth.net, optimizer_critic, gamma(steps_done)).clone().detach().to('cpu'))
                        loss_actor_dq.append(optimize_actor(batch, hero.actor.net, hero.critic.net, optimizer_actor, predictor_net).clone().detach().to('cpu'))

            soft_update(hero.critic.net, hero.critic_smooth.net)
            

            if terminated:
                if learning_circles<num_game_before_learning_game:
                    learning_circles+=1
                else: 
                    learning_circles=0
                
                avg_loss_actor = sum(loss_actor_dq)/len(loss_actor_dq)
                avg_loss_critic = sum(loss_critic_dq)/len(loss_critic_dq)
                temp = torch.tensor(loss_pred_dq)
                avg_loss_predictor = sqrt(sum(temp*temp)/len(temp))
                
                l_loss_actor.append(avg_loss_actor) # Normalisée sur le fait que la loss de l'actor donc la q value va croitre avec la valeur de gamma ((1-gamma)^-1)
                l_loss_critic.append(min(5, avg_loss_critic)) # on normalise par rapport à la valeur des score qu'il modélise. 
                l_loss_predictor.append(min(5, avg_loss_predictor))

                l_gamma.append(gamma(steps_done))
                l_epsilon.append(epsilon(steps_done))

                n_cp_hero, n_cp_adversaire = env.get_cp()

                score_hero.append(n_cp_hero)
                score_adv.append(n_cp_adversaire)

                critic_value.append(float(critic_sum)/t)
                critic_value_expected.append(float(critic_expected_sum)/t)
                hard_critic_value.append(float(hard_critic_sum)/t)
                
                
                torch.save(hero.actor.net.state_dict(), f'./models/actor/{hero.actor.version}/safetensor/{hero.actor.save_name}')
                torch.save(hero.critic.net.state_dict(), f'./models/critic/{hero.critic.version}/safetensor/{hero.critic.save_name}')
                torch.save(predictor_net.state_dict(), f'./models/predictor/{"version1"}/safetensor/test_cuda')

                plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, l_gamma, l_epsilon, eqm_critic_liste, critic_value, critic_value_expected, hard_critic_value, l_loss_predictor, l_max_loss_predictor)

                break


    print('Complete')
    plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, l_gamma, l_epsilon, eqm_critic_liste, critic_value, critic_value_expected, hard_critic_value, l_loss_predictor, l_max_loss_predictor)
    plt.ioff()
    plt.show()
    print(score_hero)