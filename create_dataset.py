import torch
import torch.optim as optim

import pickle

import environment as env

from training_utils.actor_critic import ReplayMemory, optimize_predictor

from environment.action_space import sample_action

from models.predictor.version1.architecture import Predictor



def hard_critic(arg):
    return 0

def save_dataset(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def merge_datasets(dataset_list):
    merged_dataset = []
    
    for dataset in dataset_list:
        merged_dataset.extend(dataset)
        
    return merged_dataset    

def main(max_len_normal, max_len_impact, filename_normal="dataset_normal", filename_impact="dataset_impact"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    memory = ReplayMemory(max_len_normal) # A modifier pour que ce soit adapté aux predictor (state_hero, action hero, action adv, next_state_hero)
    memory_impact = ReplayMemory(max_len_impact)
    num_episodes = 1e4
    ep=0
    while (len(memory)<max_len_normal or len(memory_impact)<max_len_impact) and ep<num_episodes:
        ep+=1
        
        observation_hero, observation_adversaire, info = env.reset()

        state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
        state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

        while True :
            
            action_hero = torch.tensor(sample_action())

            action_adv = torch.tensor(sample_action())


            observation_hero, observation_adversaire, terminated, infos_supplementaires = env.step(action_hero, action_adv)
            action_hero = action_hero.unsqueeze(0)
            action_adv = action_adv.unsqueeze(0)
            

            next_state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
            next_state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

            hard_critic_value = hard_critic(next_state_hero)
            memory.push(state_hero, action_hero, action_adv, next_state_hero, hard_critic_value)
            memory.push(state_adversaire, action_adv, action_hero, next_state_adversaire, -hard_critic_value)

            if infos_supplementaires["impact"]==True:
                memory_impact.push(state_hero, action_hero, action_adv, next_state_hero, hard_critic_value)
                memory_impact.push(state_adversaire, action_adv, action_hero, next_state_adversaire, -hard_critic_value)

            state_hero = next_state_hero
            state_adversaire = next_state_adversaire
            
            
            if terminated:
                print(ep, len(memory), len(memory_impact))
                break
        if ep%200==0:
            save_dataset(memory, filename_normal)
            save_dataset(memory_impact, filename_impact)

    print('Complete')
    
if __name__=="__main__":
    main(max_len_normal=100000, max_len_impact=5000)