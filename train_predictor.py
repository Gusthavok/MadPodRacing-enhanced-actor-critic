import matplotlib.pyplot as plt
from numpy import exp

import torch
import torch.optim as optim

import environment as env

from training_utils.joueur import Joueur, Model
from training_utils.actor_critic import ReplayMemory, optimize_predictor

from environment.action_space import sample_action

from models.predictor.version1.architecture import Predictor

def hard_critic(arg):
    return 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    memory = ReplayMemory(24000) # A modifier pour que ce soit adapté aux predictor (state_hero, action hero, action adv, next_state_hero)
    predictor_net = Predictor()
    optimizer_predictor = optim.AdamW(predictor_net.parameters(), lr=1e-3, amsgrad=True)
    
    predictor_save_name='test'
    num_episodes = 600
    num_learning_cycles = 3

    for i_episode in range(num_episodes):

        observation_hero, observation_adversaire, info = env.reset()

        state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
        state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

        avg_loss = 0
        t=-1
        while True :
            t+=1
            
            action_hero = torch.tensor(sample_action())

            action_adv = torch.tensor(sample_action())


            observation_hero, observation_adversaire, terminated, _ = env.step(action_hero, action_adv)
            action_hero = action_hero.unsqueeze(0)
            action_adv = action_adv.unsqueeze(0)
            

            next_state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
            next_state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

            hard_critic_value = hard_critic(next_state_hero)
            memory.push(state_hero, action_hero, action_adv, next_state_hero, hard_critic_value)
            memory.push(state_adversaire, action_adv, action_hero, next_state_adversaire, -hard_critic_value)

            state_hero = next_state_hero
            state_adversaire = next_state_adversaire
            
            for _ in range(num_learning_cycles):
                loss_predictor = optimize_predictor(memory, predictor_net, optimizer_predictor, BATCH_SIZE=512)
                avg_loss +=loss_predictor
            if terminated:
                
                print(f"loss moyenné sur l'épisode : {avg_loss/t}")

                torch.save(predictor_net.state_dict(), f'./models/predictor/{"version1"}/safetensor/{predictor_save_name}')

                break


    print('Complete')
    
if __name__=="__main__":
    main()