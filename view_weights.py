import dog_planning_dqn as dqn
import importlib
importlib.reload(dqn)
import torch

folder = "good_weights"
version = "8-v2"
route = f"{folder}/{version}"


device = "cuda" if torch.cuda.is_available() else "cpu"

net1 = dqn.load_weights_txt(None, f'{route}/player1_weights.txt', device=device)
net2 = dqn.load_weights_txt(None, f'{route}/player2_weights.txt', device=device)

# Derive K_dirs from the loaded net: K = K_dirs*2 + 1 (add_stay=True) → K_dirs = (K-1)//2
K_dirs = (net1.K - 1) // 2  # = 16

env = dqn.DogGame(step=dqn.STEP, K_dirs=K_dirs, add_stay=True,
                  house_r=0.03, max_episode_steps=dqn.STEPS_PER_EPISODE,
                  seed=0, houses_fixed=(0.25, 0.25, 0.75, 0.75))

dqn.browse(env, net1, net2, N=100, T=120, seed=0, device=device)