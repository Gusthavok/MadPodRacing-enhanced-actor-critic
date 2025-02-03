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

def plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, gamma_value, epsilon_value,eqm_critic_liste, critic_value, critic_value_expected, hard_critic_value, l_loss_predictor, l_max_loss_predictor, l_loss_actor_test, l_loss_critic_test, l_loss_predictor_test, show_results=False):
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
        "titre": "Actor (loss = -q)",
        "loss": l_loss_actor,
        "loss_test": l_loss_actor_test
    }
    dict22 = {
        "titre": "critic precision",
        "mean of c(s_{n+1})": critic_value, 
        "mean of c(d(s_n, a_1, a_2))": critic_value_expected,
        "MSE score predictions": eqm_critic_liste,
        "hard_critic_value(s_n)": hard_critic_value, 
    }
    dict31 = {
        "titre": "Predictor (loss = MSE)",
        "loss": l_loss_predictor,
        "loss_test": l_loss_predictor_test
        # "max_loss": l_max_loss_predictor,
    }
    dict32 = {
        "titre": "Critic (loss = MSE)",
        "loss": l_loss_critic,
        "loss_test": l_loss_critic_test
    }
    plot_on_6_diagrams(dict11, dict12, dict21, dict22, dict31, dict32)

def hard_critic(env):
    return 0

class ModelTraining:
    def __init__(self, hero:Joueur, predictor_net, optimizer_predictor, optimizer_critic, optimizer_actor):
        self.hero = hero
        self.predictor_net = predictor_net
        
        self.optimizer_predictor = optimizer_predictor
        self.optimizer_critic = optimizer_critic
        self.optimizer_actor = optimizer_actor  
        

def train_models(memory: ReplayMemory, model_training:ModelTraining, loss_pred_dq: list, loss_critic_dq: list, loss_actor_dq: list, steps_done: int, last_loss_predictor:float, last_loss_critic:float, test_dataset: bool):

    
    if len(memory) >= BATCH_SIZE:
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        loss_pred_dq.append(optimize_predictor(batch, model_training.predictor_net, model_training.optimizer_predictor, test_dataset=test_dataset).clone().detach().to('cpu'))
        loss_critic_dq.append(optimize_critic(batch, model_training.hero.critic.net, model_training.hero.critic_smooth.net, model_training.optimizer_critic, test_dataset=test_dataset, GAMMA=gamma(steps_done)).clone().detach().to('cpu'))

        if last_loss_predictor<=1*(1-random.rand()) and last_loss_critic<=5*(1-random.rand()):
            loss_actor_dq.append(optimize_actor(batch, model_training.hero.actor.net, model_training.hero.critic.net, model_training.optimizer_actor, model_training.predictor_net, test_dataset=test_dataset).clone().detach().to('cpu'))
            
            for _ in range(3): 
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                loss_actor_dq.append(optimize_actor(batch, model_training.hero.actor.net, model_training.hero.critic.net, model_training.optimizer_actor, model_training.predictor_net, test_dataset=test_dataset).clone().detach().to('cpu'))


def update_losses_values(l_loss_actor, l_loss_critic, l_loss_predictor, loss_actor_dq, loss_critic_dq, loss_pred_dq):
    avg_loss_actor = sum(loss_actor_dq)/len(loss_actor_dq)
    avg_loss_critic = sum(loss_critic_dq)/len(loss_critic_dq)
    avg_loss_predictor = sum(loss_pred_dq)/len(loss_pred_dq)
    # temp = torch.tensor(loss_pred_dq)
    # avg_loss_predictor = sqrt(sum(temp*temp)/len(temp))
    
    l_loss_actor.append(max(-100, avg_loss_actor))
    l_loss_critic.append(avg_loss_critic)
    l_loss_predictor.append( avg_loss_predictor)

def main():
    plt.ion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hero_actor_classe = Actor
    hero_actor_model_version = 'version1'
    hero_actor_save_name = 'gen_0_transformer'
    hero_actor_reload_name = 'gen_0_transformer'

    hero_critic_classe = Critic
    hero_critic_model_version = 'version1'
    hero_critic_save_name = 'gen_0_transformer'
    hero_critic_reload_name = 'gen_0_transformer'
    
    actor = Model(hero_actor_model_version, hero_actor_classe, hero_actor_save_name, hero_actor_reload_name, 'actor', device)
    critic = Model(hero_critic_model_version, hero_critic_classe, hero_critic_save_name, hero_critic_reload_name, 'critic', device) 
    critic_smooth = Model(hero_critic_model_version, hero_critic_classe, hero_critic_save_name, hero_critic_reload_name, 'critic', device)   

    predictor_net = Predictor().to(device)
    predictor_net.load_state_dict(torch.load('./models/predictor/version1/safetensor/test_cuda_normal', map_location=device))

    hero = Joueur(actor, critic, critic_smooth, eval_mode = False)

    is_adv=False
    adv = Joueur(actor, critic, critic_smooth, eval_mode = False)

    optimizer_actor = optim.AdamW(hero.actor.net.parameters(), lr=LR_ACTOR, amsgrad=True)
    optimizer_critic = optim.AdamW(hero.critic.net.parameters(), lr=LR_CRITIC, amsgrad=True)
    optimizer_predictor = optim.AdamW(predictor_net.parameters(), lr=3e-5, amsgrad=True)
    
    num_item_in_memory = 300*30
    memory = ReplayMemory(num_item_in_memory)
    memory_test = ReplayMemory(num_item_in_memory//5)
    proba_test_set = .2 #20% des observations d'une game vont dans le test set. 
    
    model_training = ModelTraining(hero=hero, predictor_net=predictor_net, optimizer_predictor=optimizer_predictor, optimizer_critic=optimizer_critic, optimizer_actor=optimizer_actor)


    steps_done = 0
    score_hero = []
    score_adv = []
    
    l_loss_actor = [0]
    l_loss_critic = [0]
    l_loss_predictor  = [0]
    l_loss_actor_test = [0]
    l_loss_critic_test = [0]
    l_loss_predictor_test  = [0]

    l_gamma, l_epsilon = [], []
    num_episodes = 10000
    
    eqm_critic_liste = []
    critic_value = []
    critic_value_expected = []
    hard_critic_value = []
    l_max_loss_predictor = []

    rate_training = .05
    memory_size_for_losses = int(300*rate_training)
    loss_actor_dq, loss_critic_dq, loss_pred_dq= deque([0], maxlen=memory_size_for_losses), deque([0], maxlen=memory_size_for_losses), deque([0], maxlen=memory_size_for_losses)
    loss_pred_test_dq, loss_critic_test_dq, loss_actor_test_dq = deque([0], maxlen=memory_size_for_losses), deque([0], maxlen=memory_size_for_losses), deque([0], maxlen=memory_size_for_losses)
    
    regenerate_dataset=0
    for i_episode in range(num_episodes):

        observation_hero, observation_adversaire, _ = env.reset()

        # print(observation_hero)
        state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
        state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

        t=-1
        critic_sum, critic_expected_sum, eqm_critic_sum, hard_critic_sum = 0, 0, 0, 0
        
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

            if random.rand()<proba_test_set:
                memory_test.push(state_hero, action_hero, action_adv, next_state_hero, state_hero[:, -1])
            else:
                memory.push(state_hero, action_hero, action_adv, next_state_hero, state_hero[:, -1])
            # memory.push(state_adversaire, action_adv, action_hero, next_state_adversaire, next_state_adversaire[:, -1])

            predictided_future_reward = hero.critic.net(next_state_hero).clone().detach()
            future_reward = hero.critic.net(torch.cat((state_hero[:, :20], predictor_net(state_hero, action_hero, action_adv)), dim=1)).clone().detach()

            critic_sum += predictided_future_reward
            critic_expected_sum += future_reward
            eqm_critic_sum += (predictided_future_reward-future_reward)**2
            hard_critic_sum += next_state_hero[0, -1]
            
            state_hero = next_state_hero
            state_adversaire = next_state_adversaire

            if regenerate_dataset==0 and random.rand()<rate_training:
                train_models(memory=memory, 
                             model_training=model_training, 
                             loss_pred_dq=loss_pred_dq, 
                             loss_critic_dq=loss_critic_dq, 
                             loss_actor_dq=loss_actor_dq, 
                             steps_done=steps_done, 
                             last_loss_predictor=l_loss_predictor[-1], 
                             last_loss_critic=l_loss_critic[-1], 
                             test_dataset=False)
                train_models(memory=memory_test, 
                             model_training=model_training, 
                             loss_pred_dq=loss_pred_test_dq, 
                             loss_critic_dq=loss_critic_test_dq, 
                             loss_actor_dq=loss_actor_test_dq, 
                             steps_done=steps_done, 
                             last_loss_predictor=l_loss_predictor[-1], 
                             last_loss_critic=l_loss_critic[-1], 
                             test_dataset=True)
            soft_update(hero.critic.net, hero.critic_smooth.net)
            

            if terminated:  
                if eqm_critic_sum>1000:
                    eqm_critic_sum=1000
            

                if regenerate_dataset>0: 
                    regenerate_dataset-=1
                else:
                    update_losses_values(l_loss_actor, l_loss_critic, l_loss_predictor, loss_actor_dq, loss_critic_dq, loss_pred_dq)
                    update_losses_values(l_loss_actor_test, l_loss_critic_test, l_loss_predictor_test, loss_actor_test_dq, loss_critic_test_dq, loss_pred_test_dq)
                    
                l_gamma.append(gamma(steps_done))
                l_epsilon.append(epsilon(steps_done))

                n_cp_hero, n_cp_adversaire = env.get_cp()

                score_hero.append(n_cp_hero)
                score_adv.append(n_cp_adversaire)

                critic_value.append(float(critic_sum)/t)
                critic_value_expected.append(float(critic_expected_sum)/t)
                eqm_critic_liste.append(float(eqm_critic_sum)/t)
                hard_critic_value.append(float(hard_critic_sum)/t)
                
                
                torch.save(hero.actor.net.state_dict(), f'./models/actor/{hero.actor.version}/safetensor/{hero.actor.save_name}')
                torch.save(hero.critic.net.state_dict(), f'./models/critic/{hero.critic.version}/safetensor/{hero.critic.save_name}')
                torch.save(predictor_net.state_dict(), f'./models/predictor/{"version1"}/safetensor/test_cuda_normal')

                plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, l_gamma, l_epsilon, eqm_critic_liste, critic_value, critic_value_expected, hard_critic_value, l_loss_predictor, l_max_loss_predictor, l_loss_actor_test, l_loss_critic_test, l_loss_predictor_test)

                break


    print('Complete')
    # plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, l_gamma, l_epsilon, eqm_critic_liste, critic_value, critic_value_expected, hard_critic_value, l_loss_predictor, l_max_loss_predictor, l_loss_actor_test, l_loss_critic_test, l_loss_predictor_test)
    plt.ioff()
    plt.show()
    print(score_hero)

if __name__ == "__main__":
    main()