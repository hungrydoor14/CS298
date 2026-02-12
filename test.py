import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Hyperparameters
# ======================
GAMMA = 0.9
GRID_SIZE = 10

CRASH_PENALTY = -10.0
STAY_PENALTY = -5.0
LIVING_COST = 1.0

ACTIONS = ['U', 'D', 'L', 'R']
A = list(range(4))


# ======================
# State encoding
# ======================
def encode_state(s):
    return torch.tensor(
        [
            s[0] / (GRID_SIZE - 1),
            s[1] / (GRID_SIZE - 1),
            s[2] / (GRID_SIZE - 1),
            s[3] / (GRID_SIZE - 1),
        ],
        dtype=torch.float32,
        device=device
    )


# ======================
# DQN
# ======================
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# Grid reward
# ======================
def make_grid_reward(n, R_max=5.0, alpha=1.0):
    c = (n - 1) / 2
    grid = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            dist = abs(x - c) + abs(y - c)
            grid[x, y] = R_max - alpha * dist
    return grid


# ======================
# Environment (UNCHANGED)
# ======================
class CarGame:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid_reward = make_grid_reward(grid_size)
        self.states = [
            (x1, y1, x2, y2)
            for x1 in range(grid_size)
            for y1 in range(grid_size)
            for x2 in range(grid_size)
            for y2 in range(grid_size)
        ]

    def move(self, x, y, a):
        if a == 0: y = min(self.grid_size - 1, y + 1)
        elif a == 1: y = max(0, y - 1)
        elif a == 2: x = max(0, x - 1)
        elif a == 3: x = min(self.grid_size - 1, x + 1)
        return x, y

    def transition(self, s, a1, a2):
        x1, y1, x2, y2 = s
        x1, y1 = self.move(x1, y1, a1)
        x2, y2 = self.move(x2, y2, a2)
        return (x1, y1, x2, y2)

    def reward(self, s, a1, a2):
        x1, y1, x2, y2 = s
        x1n, y1n, x2n, y2n = self.transition(s, a1, a2)

        if (x1n, y1n) == (x2n, y2n):
            return CRASH_PENALTY

        r = self.grid_reward[x1n, y1n]
        r -= LIVING_COST

        if (x1n, y1n) == (x1, y1):
            r += STAY_PENALTY
        if (x2n, y2n) == (x2, y2):
            r -= STAY_PENALTY

        return r


# ======================
# Replay Buffer
# ======================
class ReplayBuffer:
    def __init__(self, cap=10000):
        self.buf = deque(maxlen=cap)

    def push(self, *args):
        self.buf.append(args)

    def sample(self, n):
        batch = random.sample(self.buf, n)
        return zip(*batch)

    def __len__(self):
        return len(self.buf)


# ======================
# Minimax step (KEY PART)
# ======================
def dqn_step(env, net, s, a1, eps=0.1):
    if np.random.rand() < eps:
        a2 = np.random.choice(A)
    else:
        with torch.no_grad():
            q = net(encode_state(s)).cpu().numpy()
        a2 = np.argmin(q)

    r = env.reward(s, a1, a2)
    s_next = env.transition(s, a1, a2)
    done = (s_next[0], s_next[1]) == (s_next[2], s_next[3])
    return s_next, r, done


# ======================
# Train DQN
# ======================
def train_dqn(env, episodes=3000):
    net = DQN().to(device)
    target = DQN().to(device)
    target.load_state_dict(net.state_dict())
    target.eval()

    buf = ReplayBuffer()
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    EPS_START, EPS_END, EPS_DECAY = 0.3, 0.05, 2000
    losses = []

    for ep in range(episodes):
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-ep / EPS_DECAY)
        s = random.choice(env.states)
        ep_loss = 0.0

        for _ in range(30):
            if np.random.rand() < eps:
                a1 = np.random.choice(A)
            else:
                with torch.no_grad():
                    a1 = torch.argmax(net(encode_state(s))).item()

            s2, r, done = dqn_step(env, net, s, a1)
            buf.push(s, a1, r, s2, done)
            s = s2

            if len(buf) >= 32:
                S, A1, R, S2, D = buf.sample(32)
                S = torch.stack([encode_state(x) for x in S])
                A1 = torch.tensor(A1, device=device)
                R = torch.tensor(R, device=device, dtype=torch.float32)
                S2 = torch.stack([encode_state(x) for x in S2])
                D = torch.tensor(D, device=device, dtype=torch.float32)

                q = net(S).gather(1, A1.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    v_next = torch.min(target(S2), dim=1)[0]
                    tgt = R + GAMMA * v_next * (1 - D)

                loss = loss_fn(q, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()

            if done:
                break

        losses.append(ep_loss)

        if ep % 50 == 0:
            target.load_state_dict(net.state_dict())
            print(f"Episode {ep}, loss = {ep_loss:.3f}")

    return net, losses


# ======================
# DQN → policy dict (LIKE Pi1 / Pi2)
# ======================
def dqn_policy(net, env):
    policy = {}
    for s in env.states:
        def p1(s=s):
            with torch.no_grad():
                q = net(encode_state(s)).cpu().numpy()
            return np.argmax(q)

        def p2(s=s):
            with torch.no_grad():
                q = net(encode_state(s)).cpu().numpy()
            return np.argmin(q)

        policy[s] = (p1, p2)
    return policy


# ======================
# Rollout + drawing (UNCHANGED)
# ======================
def rollout(env, s0, policy, T=30):
    traj = [s0]
    s = s0
    for _ in range(T):
        a1 = policy[s][0]()
        a2 = policy[s][1]()
        s = env.transition(s, a1, a2)
        traj.append(s)
    return traj


def draw_trajectory(ax, traj, grid_size, subtitle=""):
    ax.clear()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_title(subtitle, fontsize=10)

    offset = 0.12
    off1 = np.array([ offset,  offset])
    off2 = np.array([-offset, -offset])

    for s, s_next in zip(traj[:-1], traj[1:]):
        p1s = np.array([s[0]+0.5, s[1]+0.5]) + off1
        p1e = np.array([s_next[0]+0.5, s_next[1]+0.5]) + off1
        p2s = np.array([s[2]+0.5, s[3]+0.5]) + off2
        p2e = np.array([s_next[2]+0.5, s_next[3]+0.5]) + off2

        ax.arrow(p1s[0], p1s[1], *(p1e-p1s), color="red", head_width=0.15)
        ax.arrow(p2s[0], p2s[1], *(p2e-p2s), color="blue", head_width=0.15)


def trajectory_stats(traj):
    p1 = p2 = 0
    for s, s2 in zip(traj[:-1], traj[1:]):
        if (s[0], s[1]) != (s2[0], s2[1]): p1 += 1
        if (s[2], s[3]) != (s2[2], s2[3]): p2 += 1
    return len(traj)-1, p1, p2, len(set(traj))


# ======================
# RUN + INTERACTIVE VIEWER (SAME AS BEFORE)
# ======================
if __name__ == "__main__":
    env = CarGame(GRID_SIZE)
    net, losses = train_dqn(env)
    policy = dqn_policy(net, env)

    states = env.states
    trajectories = [rollout(env, s, policy, T=20) for s in states]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    idx = [0]

    def update():
        s = states[idx[0]]
        traj = trajectories[idx[0]]
        total, p1, p2, uniq = trajectory_stats(traj)

        subtitle = f"Moves: {total} | P1: {p1} | P2: {p2} | Unique: {uniq}"
        draw_trajectory(ax, traj, env.grid_size, subtitle)
        fig.suptitle(f"DQN trajectory from {s}", fontsize=14)
        fig.canvas.draw_idle()

    def next_state(event):
        idx[0] = (idx[0] + 1) % len(states)
        update()

    def prev_state(event):
        idx[0] = (idx[0] - 1) % len(states)
        update()

    axprev = plt.axes([0.25, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.65, 0.05, 0.1, 0.075])

    Button(axprev, "Prev").on_clicked(prev_state)
    Button(axnext, "Next").on_clicked(next_state)

    update()
    plt.show()

    plt.figure()
    plt.plot(losses)
    plt.title("DQN Training Loss")
    plt.show()
