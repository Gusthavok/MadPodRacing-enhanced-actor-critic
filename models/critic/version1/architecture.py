import torch
import torch.nn as nn
import torch.nn.functional as F


# class Critic(nn.Module):
#     def __init__(self, hidden_dimension=64, num_layers=2, num_heads=4):
#         super(Critic, self).__init__()
        
#         self.embedding = nn.Linear(55, hidden_dimension)  # Projection des entrées
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dimension, 
#             nhead=num_heads, 
#             dim_feedforward=hidden_dimension * 4,
#             activation="gelu"
#         )
        
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.final_layer = nn.Linear(hidden_dimension, 1)  # Sortie avec une seule valeur

#     def forward(self, observation):
#         x = self.embedding(observation)  # Projection initiale
#         x = x.unsqueeze(0)  # Ajouter une dimension de batch pour le transformer
#         x = self.transformer_encoder(x)  # Passage dans le Transformer
#         x = x.squeeze(0)  # Enlever la dimension batch
#         return self.final_layer(x)


class Critic(nn.Module):

    def __init__(self, hidden_dimension=64):
        super(Critic, self).__init__()

        # Première couche : après produit extérieur, entrée avec 3025 paramètres
        self.first_layer = nn.Linear(55 * 55 + 55, hidden_dimension)  # 55x55 = 3025

        # Couches cachées (1 seule couche ici mais peut être plus)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1)])

        # Dernière couche pour produire l'output
        self.final_layer = nn.Linear(hidden_dimension, 1)
    
    def forward(self, observation):
        # Calcul du produit extérieur des features pour chaque élément du batch
        batch_size, num_features = observation.shape

        # On crée un produit extérieur (batch_size, 55, 55) pour chaque entrée
        interaction_matrix = torch.bmm(observation.unsqueeze(2), observation.unsqueeze(1))  # (batch_size, 55, 55)

        # Maintenant on aplatie la matrice d'interaction pour l'entrer dans le réseau
        interaction_matrix_flat = interaction_matrix.flatten(start_dim=1)  # (batch_size, 55*55)

        # Passer la matrice d'interaction aplatie à travers la première couche
        x = torch.cat((interaction_matrix_flat, observation), dim=1)
        x = self.first_layer(x)  # Cette couche attend maintenant 3025 entrées

        # Couches cachées
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))

        # Dernière couche pour produire le résultat
        x = self.final_layer(x)

        return x



# class Critic(nn.Module):

#     def __init__(self, hidden_dimension=64):
#         super(Critic, self).__init__()
    

#         self.first_layer = nn.Linear(55, hidden_dimension) # 55 informations de l'environnement 
        
#         self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1)])

#         self.final_layer = nn.Linear(hidden_dimension, 1)

    
#     def forward(self, observation):
#         x = observation
#         x = self.first_layer(x) # action admet 50 paramètres        
 

#         for layer in self.hidden_layers:
#             x = F.leaky_relu(layer(x))

#         return self.final_layer(x)