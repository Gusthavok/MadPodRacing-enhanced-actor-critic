import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, hidden_dimension=64):
        super(Critic, self).__init__()
    

        self.first_layer = nn.Linear(55, hidden_dimension) # 55 informations de l'environnement 
        
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1)])

        self.final_layer = nn.Linear(hidden_dimension, 1)

    
    def forward(self, observation):
        x = observation
        x = self.first_layer(x) # action admet 50 paramètres        
 

        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))

        return self.final_layer(x)