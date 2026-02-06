import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.9
TOL = 1e-3
MAX_Q_ITERS = 200
FP_ITERS = 800

GRID_SIZE = 3

CRASH_PENALTY = -10.0
STAY_PENALTY = -5.0
LIVING_COST = 1.0

# Actions: Up, Down, Left, Right
ACTIONS = ['U', 'D', 'L', 'R']
A = list(range(len(ACTIONS)))  # [0, 1, 2, 3]

def encode_state(s, grid_size):
    """
    s = (x1, y1, x2, y2)
    returns torch.FloatTensor shape (4,)
    """
    return torch.tensor(
        [
            s[0] / (grid_size - 1),
            s[1] / (grid_size - 1),
            s[2] / (grid_size - 1),
            s[3] / (grid_size - 1),
        ],
        dtype=torch.float32,
        device=device
    )

class DQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, s):
        return self.net(s)

def opponent_policy(s):
    return np.random.choice(A)

def evaluate_policy(env, net, episodes=50):
    lengths = []
    for _ in range(episodes):
        s = random.choice(env.states)
        for t in range(30):
            with torch.no_grad():
                a = torch.argmax(net(encode_state(s, env.grid_size))).item()
            s, _, done = dqn_step(env, s, a)
            if done:
                break
        lengths.append(t+1)
    return np.mean(lengths)

def dqn_step(env, s, a1):
    a2 = opponent_policy(s)
    r = env.reward(s, a1, a2)
    s_next = env.transition(s, a1, a2)

    done = (s_next[0], s_next[1]) == (s_next[2], s_next[3])
    return s_next, r, done

def select_action(net, s_tensor, eps):
    if np.random.rand() < eps:
        return np.random.choice(len(A))
    with torch.no_grad():
        return torch.argmax(net(s_tensor)).item()
    
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buffer)

def train_dqn(env, episodes=3000, T=30):
    net = DQN().to(device)
    buffer = ReplayBuffer()
    BATCH_SIZE = 32

    # TARGET NETWORK (freeze copy)
    target_net = DQN().to(device)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    TARGET_UPDATE = 50  # sync every 50 episodes

    GAMMA = 0.9
    EPS_START = 0.3
    EPS_END = 0.05
    EPS_DECAY = 2000

    losses = []

    for ep in range(episodes):
        EPS = EPS_END + (EPS_START - EPS_END) * np.exp(-ep / EPS_DECAY)
        s = random.choice(env.states)

        ep_loss = 0.0

        for t in range(T):
            s_tensor = encode_state(s, env.grid_size)

            a = select_action(net, s_tensor, EPS)

            s_next, r, done = dqn_step(env, s, a)
            buffer.push(s, a, r, s_next, done)
            s_next_tensor = encode_state(s_next, env.grid_size)

            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

                s_batch = torch.stack([encode_state(x, env.grid_size) for x in states])
                a_batch = torch.tensor(actions, device=device)
                r_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
                sn_batch = torch.stack([encode_state(x, env.grid_size) for x in next_states])
                done_batch = torch.tensor(dones, dtype=torch.float32, device=device)

                q_vals = net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Minimax continuation value
                    next_q = target_net(sn_batch)          # shape [B, |A|]
                    v_next = torch.min(next_q, dim=1)[0]   # minimax value
                    target_vals = r_batch + GAMMA * v_next * (1 - done_batch)

                loss = loss_fn(q_vals, target_vals)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()

            if len(buffer) >= BATCH_SIZE:
                ep_loss += loss.item()

            s = s_next

            if done:
                break

        losses.append(ep_loss / (t + 1))

        if ep % 100 == 0:
            avg_len = evaluate_policy(env, net, episodes=20)
            print(
                f"Episode {ep}, steps = {t+1}, "
                f"avg loss = {losses[-1]:.3f}, "
                f"eval length = {avg_len:.2f}"
            )

        if ep > 0 and ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(net.state_dict())

    return net, losses

def dqn_policy(net, env, eps=0.0):
    """
    Wraps a trained DQN into a (pi1, pi2) policy
    compatible with rollout().
    """
    policy = {}

    for s in env.states:
        s_tensor = encode_state(s, env.grid_size)

        with torch.no_grad():
            q_vals = net(s_tensor).cpu().numpy()

        # Player 1: greedy (or epsilon-greedy if you want)
        def p1_action(q=q_vals):
            if np.random.rand() < eps:
                return np.random.choice(len(A))
            return np.argmax(q)

        # Player 2: worst-case adversary
        def p2_action(q=q_vals):
            return np.argmin(q)


        policy[s] = (p1_action, p2_action)

    return policy


def make_grid_reward(n, R_max=5.0, alpha=1.0):
    c = (n - 1) / 2
    grid = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            dist = abs(x - c) + abs(y - c)
            grid[x, y] = R_max - alpha * dist
    return grid

class CarGame:
    """
    Two-player zero-sum grid game.
    State: (x1, y1, x2, y2)
    Player 1 tries to collide with Player 2.
    """

    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.states = self._all_states()

        self.grid_reward = make_grid_reward(grid_size, R_max=5.0, alpha=1.0)


    def _all_states(self):
        S = []
        # Enumerate all possible joint positions
        for x1 in range(self.grid_size):
            for y1 in range(self.grid_size):
                for x2 in range(self.grid_size):
                    for y2 in range(self.grid_size):
                        S.append((x1, y1, x2, y2))
        return S

    def move(self, x, y, a):
        # Apply grid movement with boundary conditions
        if a == 0:   # U
            y = min(self.grid_size - 1, y + 1)
        elif a == 1: # D
            y = max(0, y - 1)
        elif a == 2: # L
            x = max(0, x - 1)
        elif a == 3: # R
            x = min(self.grid_size - 1, x + 1)
        return x, y

    def transition(self, s, a1, a2):
        x1, y1, x2, y2 = s
        x1, y1 = self.move(x1, y1, a1)
        x2, y2 = self.move(x2, y2, a2)
        return (x1, y1, x2, y2)

    def reward(self, s, a1, a2):
        r = 0.0 # total reward

        # current state
        x1, y1, x2, y2 = s
        # next state
        x1n, y1n, x2n, y2n = self.transition(s, a1, a2)

        # crash
        if (x1n, y1n) == (x2n, y2n):
            return CRASH_PENALTY
        
        # grid reward (player 1 perspective)
        r += self.grid_reward[x1n, y1n]

        # living cost
        r -= LIVING_COST

        # penalize staying in place
        if (x1n, y1n) == (x1, y1):
            r += STAY_PENALTY

        if (x2n, y2n) == (x2, y2):
            r -= STAY_PENALTY   # zero-sum symmetry

        return r
    
def rollout(env, s0, policy, T=30):
    traj = [s0]
    s = s0
    for _ in range(T):
        a1 = policy[s][0]()
        a2 = policy[s][1]()
        s = env.transition(s, a1, a2)
        traj.append(s)
    return traj

def draw_trajectory(ax, traj, grid_size, title="", subtitle=""):
    ax.clear()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_title(subtitle, fontsize=10, color="gray", pad=6)

    offset = 0.12
    off1 = np.array([ offset,  offset])   # Player 1 (red)
    off2 = np.array([-offset, -offset])   # Player 2 (blue)

    for s, s_next in zip(traj[:-1], traj[1:]):

        # Player 1
        p1_start = np.array([s[0] + 0.5, s[1] + 0.5]) + off1
        p1_end   = np.array([s_next[0] + 0.5, s_next[1] + 0.5]) + off1
        d1 = p1_end - p1_start

        if np.linalg.norm(d1) > 1e-6:
            ax.arrow(
                p1_start[0], p1_start[1],
                d1[0], d1[1],
                color="red",
                head_width=0.15,
                length_includes_head=True
            )

        # Player 2
        p2_start = np.array([s[2] + 0.5, s[3] + 0.5]) + off2
        p2_end   = np.array([s_next[2] + 0.5, s_next[3] + 0.5]) + off2
        d2 = p2_end - p2_start

        if np.linalg.norm(d2) > 1e-6:
            ax.arrow(
                p2_start[0], p2_start[1],
                d2[0], d2[1],
                color="blue",
                head_width=0.15,
                length_includes_head=True
            )

def trajectory_stats(traj):
    p1_moves = 0
    p2_moves = 0

    for s, s_next in zip(traj[:-1], traj[1:]):
        if (s[0], s[1]) != (s_next[0], s_next[1]):
            p1_moves += 1
        if (s[2], s[3]) != (s_next[2], s_next[3]):
            p2_moves += 1

    total_moves = len(traj) - 1
    unique_states = len(set(traj))

    return total_moves, p1_moves, p2_moves, unique_states

if __name__ == "__main__":
    env = CarGame(grid_size=GRID_SIZE)

    # ---- TRAIN ----
    dqn, losses = train_dqn(env)

    avg_len = evaluate_policy(env, dqn, episodes=100)
    print(f"Final greedy policy avg episode length: {avg_len:.2f}")

    # ---- BUILD DQN ROLLOUTS FOR ALL STATES ----
    policy = dqn_policy(dqn, env)

    states = env.states
    trajectories = [
        rollout(env, s, policy, T=20)
        for s in states
    ]

    # ---- INTERACTIVE VIEWER ----
    from matplotlib.widgets import Button

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    idx = [0]  # mutable index

    def update():
        s = states[idx[0]]
        traj = trajectories[idx[0]]

        total, p1, p2, uniq = trajectory_stats(traj)

        subtitle = (
            f"Total moves: {total} | "
            f"P1 moves: {p1} | "
            f"P2 moves: {p2} | "
            f"Unique states: {uniq}"
        )

        draw_trajectory(
            ax,
            traj,
            env.grid_size,
            subtitle=subtitle
        )

        fig.suptitle(
            f"DQN rollout from {s}",
            fontsize=16,
            y=0.97
        )

        fig.canvas.draw_idle()

    def next_state(event):
        idx[0] = (idx[0] + 1) % len(states)
        update()

    def prev_state(event):
        idx[0] = (idx[0] - 1) % len(states)
        update()

    axprev = plt.axes([0.25, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.65, 0.05, 0.1, 0.075])

    bprev = Button(axprev, "Prev")
    bnext = Button(axnext, "Next")

    bprev.on_clicked(prev_state)
    bnext.on_clicked(next_state)

    update()
    plt.show()

    # ---- LOSS CURVE ----
    plt.figure()
    plt.plot(losses)
    plt.title("DQN training loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()
