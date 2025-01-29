 
import torch
import torch.optim as optim

import environment as env

from training_utils.actor_critic import ReplayMemory, optimize_predictor, Transition

from environment.action_space import sample_action

from models.predictor.version1.architecture import Predictor

def hard_critic(arg):
    return 0

def main(dataset_normal, dataset_impact, num_iteration_for_avg_loss = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predictor_net = Predictor().to(device)
    optimizer_predictor = optim.AdamW(predictor_net.parameters(), lr=1e-3, amsgrad=True)
    
    predictor_save_name='test_cuda'

    for ind in range(1000):

        avg_loss = 0
        avg_loss_impact = 0
        num_train_sur_impact = 1e-6
        for i in range(num_iteration_for_avg_loss):
            transitions = dataset_normal.sample(512)
            batch = Transition(*zip(*transitions))
            avg_loss += optimize_predictor(batch, predictor_net, optimizer_predictor)
            if i%100==0 and ind>10:
                transitions = dataset_impact.sample(64)
                batch = Transition(*zip(*transitions))
                avg_loss_impact += optimize_predictor(batch, predictor_net, optimizer_predictor)
                num_train_sur_impact+=1
        

        print(f'loss moyenée : {avg_loss/num_iteration_for_avg_loss} // loss sur les impacts : {avg_loss_impact/num_train_sur_impact}')
        torch.save(predictor_net.state_dict(), f'./models/predictor/{"version1"}/safetensor/{predictor_save_name}')


if __name__=="__main__":
    from create_dataset import load_dataset
    dataset_normal = load_dataset('./dataset_normal')
    dataset_impact = load_dataset('./dataset_impact')
    main(dataset_normal, dataset_impact)