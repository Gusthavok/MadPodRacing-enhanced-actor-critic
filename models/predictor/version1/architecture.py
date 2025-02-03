import torch
import torch.nn as nn
import torch.nn.functional as F

# class Predictor(nn.Module):
#     def __init__(self, hidden_dimension=128, num_layers=1, num_heads=4):
#         super(Predictor, self).__init__()
        
#         self.embedding = nn.Linear(55 + 10 + 10, hidden_dimension)  # Projection des entrées combinées
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dimension, 
#             nhead=num_heads, 
#             dim_feedforward=hidden_dimension * 4,
#             activation="gelu"
#         )
        
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.final_layer = nn.Linear(hidden_dimension, 35)  # Sortie avec 35 dimensions

#     def forward(self, observation, action_J1, action_J2):
#         x = torch.cat((observation, action_J1, action_J2), dim=1)  # Concaténation des entrées
#         x = self.embedding(x)  # Projection initiale
#         x = x.unsqueeze(0)  # Ajouter une dimension de batch pour le transformer
#         x = self.transformer_encoder(x)  # Passage dans le Transformer
#         x = x.squeeze(0)  # Enlever la dimension batch
#         return self.final_layer(x)


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