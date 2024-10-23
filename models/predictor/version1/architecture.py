import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):

    def __init__(self, hidden_dimension=128):
        super(Predictor, self).__init__()
    

        self.first_layer = nn.Linear(55+10+10, hidden_dimension) # 24+20+10+1 (on utilise les 55 pour prédire 24 + 10+1) informations de l'env, 10 informations par joueur
        
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1)])

        self.final_layer = nn.Linear(hidden_dimension, 35)

    
    def forward(self, observation, action_J1, action_J2):
        x = torch.cat((observation, action_J1, action_J2), dim=1)
        x = self.first_layer(x)        
 

        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))

        return self.final_layer(x)