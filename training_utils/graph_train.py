import torch
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

import matplotlib.pyplot as plt


def plot_on_4_diagrams(dict11, dict12, dict21, dict22, show_result=False):
    # Crée une figure avec 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Liste des dictionnaires et des sous-graphiques associés
    dicts = [(dict11, axs[0, 0]), (dict12, axs[0, 1]), 
             (dict21, axs[1, 0]), (dict22, axs[1, 1])]
    
    for d, ax in dicts:
        title = d.pop('titre')  # Récupère et retire le titre du dictionnaire
        with_mean = d.pop('with_mean', False)  # Vérifie si 'with_mean' est présent
        
        for key, value in d.items():
            ax.plot(value, label=key)
            # Si 'with_mean' est True et que la liste a plus de 100 items
            if with_mean and len(value) > 100:
                # Calcul de la moyenne des 100 dernières valeurs
                mean_last_100 = [sum(value[i-100:i])/100 for i in range(100, len(value)+1)]
                ax.plot(range(99, len(value)), mean_last_100, label=f"{key} - Mean Last 100", linestyle='--')
                
        ax.set_title(title)      # Ajoute le titre du subplot
        ax.legend()              # Ajoute la légende
    
    plt.tight_layout()
    plt.pause(0.02)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_durations(score_attaque, score_defense, show_result=False):
    plt.figure(1)
    sc_atq = torch.tensor(score_attaque, dtype=torch.float)
    sc_dfs = torch.tensor(score_defense, dtype=torch.float)
    sc_diff = sc_atq - sc_dfs

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(sc_atq.numpy(), label='hero')
    plt.plot(sc_dfs.numpy(), label='adversaire')
    plt.plot(sc_diff.numpy(), label='difference')
    # Take 100 episode averages and plot them too
    if len(sc_atq) >= 100:
        means_atq = sc_atq.unfold(0, 100, 1).mean(1).view(-1)
        means_atq = torch.cat((torch.zeros(99), means_atq))
        plt.plot(means_atq.numpy())

        means_dfs = sc_dfs.unfold(0, 100, 1).mean(1).view(-1)
        means_dfs = torch.cat((torch.zeros(99), means_dfs))
        plt.plot(means_dfs.numpy())

        means_diff = sc_diff.unfold(0, 100, 1).mean(1).view(-1)
        means_diff = torch.cat((torch.zeros(99), means_diff))
        plt.plot(means_diff.numpy())

    plt.legend(loc='lower left')

    plt.pause(0.02)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
