import torch

class Model:
    def __init__(self, model_version, classe, save_name, reload_name, n_observations, n_actions, type_model, device):
        self.net = classe(n_observations, n_actions).to(device)
        if reload_name != '':
            self.net.load_state_dict(torch.load(f'./models/{type_model}/{model_version}/safetensor/{reload_name}', map_location=device))
        self.version = model_version
        self.save_name = save_name

class Joueur:
    def __init__(self, actor : Model, critic : Model, critic_smooth : Model, eval_mode = False):
        self.actor = actor
        self.critic = critic
        self.critic_smooth = critic_smooth
        if eval_mode:
            self.actor.net.eval()
            self.critic.net.eval()
            self.critic_smooth.net.eval()

    def action(self, state):
        return self.actor.net(state).squeeze()