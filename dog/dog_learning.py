import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time

GAMMA = 0.65
STEPS = 600_000
EPS_DECAY = 300_000
LR = 1e-4

STEP = 0.05
NUM_ACTIONS = 16
STEPS_PER_EPISODE = 150
WARMUP = 5000
BATCH_SIZE = 64


#  Environment (unchanged) 
class DogGame:
    def __init__(self, step=0.02, K_dirs=8, add_stay=False, house_r=0.03,
                 max_episode_steps=150, seed=0, houses_fixed=(0.25, 0.25, 0.75, 0.75)):
        self.step = float(step)
        self.K_dirs = int(K_dirs)
        self.add_stay = bool(add_stay)
        self.house_r = float(house_r)
        self.max_episode_steps = int(max_episode_steps)
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self.houses_fixed = houses_fixed
        self.K = self.K_dirs * 2 + (1 if self.add_stay else 0)
        self.stay_idx = self.K_dirs * 2 if self.add_stay else None
        base_angles = (2.0 * np.pi) * (np.arange(self.K_dirs) / self.K_dirs)
        self.angles = base_angles
        self.action_steps = np.concatenate([
            np.full(self.K_dirs, self.step),
            np.full(self.K_dirs, self.step * 0.5),
        ])
        self.action_angles = np.concatenate([base_angles, base_angles])

    def _dist(self, ax, ay, bx, by):
        return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))

    def reset(self, s0=None):
        self._t = 0
        h1x, h1y, h2x, h2y = map(float, self.houses_fixed)
        if s0 is None:
            p = self.rng.random(4).astype(np.float32)
            return np.array([p[0], p[1], p[2], p[3], h1x, h1y, h2x, h2y], dtype=np.float32)
        s0 = np.array(s0, dtype=np.float32)
        if s0.shape[0] == 4:
            return np.array([s0[0], s0[1], s0[2], s0[3], h1x, h1y, h2x, h2y], dtype=np.float32)
        return s0

    def _move(self, x, y, a):
        a = int(a)
        if self.add_stay and a == self.stay_idx:
            return x, y
        theta = float(self.action_angles[a])
        step  = float(self.action_steps[a])
        x2 = float(np.clip(x + step * np.cos(theta), 0.0, 1.0))
        y2 = float(np.clip(y + step * np.sin(theta), 0.0, 1.0))
        return x2, y2

    def step_env(self, s, a1, a2):
        self._t += 1
        x1, y1, x2, y2, h1x, h1y, h2x, h2y = map(float, s)
        x1n, y1n = self._move(x1, y1, a1)
        x2n, y2n = self._move(x2, y2, a2)
        dogx = 0.5 * (x1n + x2n)
        dogy = 0.5 * (y1n + y2n)
        d1 = self._dist(dogx, dogy, h1x, h1y)
        d2 = self._dist(dogx, dogy, h2x, h2y)
        r1 = -d1
        r2 = -d2
        done = (self._t >= self.max_episode_steps)
        sn = np.array([x1n, y1n, x2n, y2n, h1x, h1y, h2x, h2y], dtype=np.float32)
        return sn, r1, r2, done


#  Networks 
# Each agent has its own Q-network that maps state -> Q(s, a) for each of its K actions.
# No joint action matrix needed.

class QNet(nn.Module):
    def __init__(self, state_dim=8, K=17, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, K),         # K values, one per own action
        )

    def forward(self, s):
        return self.net(s)  # (batch, K)


# Replay buffer (unchanged) 

class Replay:
    def __init__(self, cap=100_000):
        self.buf = deque(maxlen=cap)

    def add(self, s, a1, a2, r1, r2, sn, done):
        self.buf.append((s, a1, a2, r1, r2, sn, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a1, a2, r1, r2, sn, done = zip(*batch)
        return (np.stack(s), np.array(a1, dtype=np.int64), np.array(a2, dtype=np.int64),
                np.array(r1, dtype=np.float32), np.array(r2, dtype=np.float32),
                np.stack(sn), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


#  Helpers 

@torch.no_grad()
def soft_update(tgt, src, tau=0.005):
    for pt, ps in zip(tgt.parameters(), src.parameters()):
        pt.data.mul_(1 - tau).add_(tau * ps.data)


@torch.no_grad()
def greedy_actions(net1, net2, s_np, device):
    """Pure greedy action selection (used at eval/rollout time)."""
    s = torch.tensor(s_np[None], dtype=torch.float32, device=device)
    a1 = int(net1(s).argmax(dim=1).item())
    a2 = int(net2(s).argmax(dim=1).item())
    return a1, a2


def eps_greedy(net, s_np, K, eps, rng, device):
    """Epsilon-greedy for a single agent."""
    if rng.random() < eps:
        return int(rng.integers(K))
    s = torch.tensor(s_np[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        return int(net(s).argmax(dim=1).item())


# Training 

def train(env, steps=STEPS, warmup=WARMUP, batch_size=BATCH_SIZE, lr=LR,
          eps_start=1.0, eps_end=0.05, eps_decay_steps=EPS_DECAY,
          target_sync=100, grad_clip=5.0, hidden=256, seed=0, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    K = env.K

    # Each agent: online net + target net
    net1  = QNet(state_dim=8, K=K, hidden=hidden).to(device)
    net2  = QNet(state_dim=8, K=K, hidden=hidden).to(device)
    tgt1  = QNet(state_dim=8, K=K, hidden=hidden).to(device)
    tgt2  = QNet(state_dim=8, K=K, hidden=hidden).to(device)
    tgt1.load_state_dict(net1.state_dict()); tgt1.eval()
    tgt2.load_state_dict(net2.state_dict()); tgt2.eval()

    opt1 = optim.Adam(net1.parameters(), lr=lr)
    opt2 = optim.Adam(net2.parameters(), lr=lr)

    replay = Replay()
    rng = np.random.default_rng(seed + 1)

    s = env.reset()
    loss1_history, loss2_history = [], []

    for t in range(1, steps + 1):
        #  Epsilon schedule 
        frac = min(1.0, t / eps_decay_steps)
        eps  = eps_start + frac * (eps_end - eps_start)

        #  Action selection: epsilon-greedy for BOTH agents 
        a1 = eps_greedy(net1, s, K, eps, rng, device)
        a2 = eps_greedy(net2, s, K, eps, rng, device)

        sn, r1, r2, done = env.step_env(s, a1, a2)
        replay.add(s, a1, a2, r1, r2, sn, done)
        s = env.reset() if done else sn

        if len(replay) < warmup:
            continue

        #  Sample batch 
        sb, a1b, a2b, r1b, r2b, snb, doneb = replay.sample(batch_size)

        sb_t   = torch.tensor(sb,    dtype=torch.float32, device=device)
        snb_t  = torch.tensor(snb,   dtype=torch.float32, device=device)
        r1b_t  = torch.tensor(r1b,   dtype=torch.float32, device=device)
        r2b_t  = torch.tensor(r2b,   dtype=torch.float32, device=device)
        done_t = torch.tensor(doneb, dtype=torch.float32, device=device)
        a1b_t  = torch.tensor(a1b,   dtype=torch.int64,   device=device)
        a2b_t  = torch.tensor(a2b,   dtype=torch.int64,   device=device)

        # ── TD targets: standard Bellman backup, each agent independently ─
        with torch.no_grad():
            v1_next = tgt1(snb_t).max(dim=1).values   # max over own actions
            v2_next = tgt2(snb_t).max(dim=1).values

        y1 = r1b_t + (1.0 - done_t) * GAMMA * v1_next
        y2 = r2b_t + (1.0 - done_t) * GAMMA * v2_next

        #  Update agent 1 
        Q1_sa  = net1(sb_t).gather(1, a1b_t.unsqueeze(1)).squeeze(1)
        loss1  = nn.functional.smooth_l1_loss(Q1_sa, y1)
        opt1.zero_grad(); loss1.backward()
        nn.utils.clip_grad_norm_(net1.parameters(), grad_clip)
        opt1.step()

        #  Update agent 2 
        Q2_sa  = net2(sb_t).gather(1, a2b_t.unsqueeze(1)).squeeze(1)
        loss2  = nn.functional.smooth_l1_loss(Q2_sa, y2)
        opt2.zero_grad(); loss2.backward()
        nn.utils.clip_grad_norm_(net2.parameters(), grad_clip)
        opt2.step()

        #  Soft-update target nets 
        if t % target_sync == 0:
            soft_update(tgt1, net1)
            soft_update(tgt2, net2)

        loss1_history.append(float(loss1))
        loss2_history.append(float(loss2))

        if t % 500 == 0:
            print(f"step {t:6d}  eps {eps:.3f}  "
                  f"loss1 {float(loss1):.5f}  loss2 {float(loss2):.5f}  "
                  f"replay {len(replay)}")

    return net1, net2, loss1_history, loss2_history


#  Visualisation (unchanged except using greedy_actions) 

def plot_loss(loss1_history, loss2_history, window=500):
    fig, ax = plt.subplots(figsize=(10, 4))
    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")
    steps = np.arange(len(loss1_history))
    ax.plot(steps, loss1_history, color="blue", alpha=0.15)
    ax.plot(steps, loss2_history, color="red",  alpha=0.15)
    if len(loss1_history) >= window:
        s = smooth(loss1_history, window)
        ax.plot(np.arange(len(s)) + window // 2, s, color="blue", lw=2, label="Player 1 (smoothed)")
        s = smooth(loss2_history, window)
        ax.plot(np.arange(len(s)) + window // 2, s, color="red",  lw=2, label="Player 2 (smoothed)")
    ax.set_xlabel("Training step"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.set_yscale("log")
    plt.tight_layout(); plt.show()


def rollout(env, net1, net2, s0, T=100, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    s = env.reset(s0=s0)
    traj = [s.copy()]
    r1s, r2s = [], []
    for _ in range(T):
        a1, a2 = greedy_actions(net1, net2, s, device)
        sn, r1, r2, done = env.step_env(s, a1, a2)
        traj.append(sn.copy()); r1s.append(r1); r2s.append(r2)
        s = sn
        if done:
            break
    return traj, r1s, r2s


def draw_traj(ax, traj, env, title=None):
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    if title: ax.set_title(title, fontsize=9)
    h1x, h1y, h2x, h2y = map(float, traj[0][4:])
    ax.scatter([h1x], [h1y], marker="s", s=200, color="blue", zorder=5)
    ax.scatter([h2x], [h2y], marker="s", s=200, color="red",  zorder=5)
    arr = np.array(traj); p1 = arr[:, :2]; p2 = arr[:, 2:4]; dog = 0.5 * (p1 + p2)
    hw = 0.012
    for k in range(len(traj) - 1):
        for pts, col in [(p1, "blue"), (p2, "red"), (dog, "black")]:
            dx, dy = pts[k+1] - pts[k]
            if abs(dx) + abs(dy) > 1e-6:
                ax.arrow(pts[k,0], pts[k,1], dx, dy, color=col, head_width=hw,
                         length_includes_head=True, zorder=3)
    ax.scatter(*p1[0],  marker="o", s=60,  color="blue",  zorder=6)
    ax.scatter(*p2[0],  marker="o", s=60,  color="red",   zorder=6)
    ax.scatter(*dog[0], marker="x", s=100, color="black", zorder=6)


@torch.no_grad()
def draw_policy_field(ax_blue, ax_red, env, net1, net2, device, grid_n=15):
    h1x, h1y, h2x, h2y = map(float, env.houses_fixed)
    lin = np.linspace(0.05, 0.95, grid_n)
    xs, ys = np.meshgrid(lin, lin)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    u1s, v1s, u2s, v2s = [], [], [], []
    for px, py in pts:
        s = np.array([px, py, 1-px, 1-py, h1x, h1y, h2x, h2y], dtype=np.float32)
        s_t = torch.tensor(s[None], dtype=torch.float32, device=device)
        a1 = int(net1(s_t).argmax(dim=1).item())
        a2 = int(net2(s_t).argmax(dim=1).item())
        for a, us, vs in [(a1, u1s, v1s), (a2, u2s, v2s)]:
            if env.add_stay and a == env.stay_idx:
                us.append(0.0); vs.append(0.0)
            else:
                theta = float(env.action_angles[a])
                us.append(np.cos(theta)); vs.append(np.sin(theta))
    scale = 0.9 / grid_n
    for ax, color, us, vs_arr, title in [
        (ax_blue, "blue", np.array(u1s), np.array(v1s), "Blue policy"),
        (ax_red,  "red",  np.array(u2s), np.array(v2s), "Red policy"),
    ]:
        ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.scatter([h1x], [h1y], marker="s", s=150, color="blue", zorder=5)
        ax.scatter([h2x], [h2y], marker="s", s=150, color="red",  zorder=5)
        ax.quiver(pts[:,0], pts[:,1], us * scale, vs_arr * scale,
                  color=color, alpha=0.8, angles="xy", scale_units="xy",
                  scale=1, width=0.004, headwidth=4)


def browse(env, net1, net2, N=100, T=120, seed=0, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    starts = [env.reset() for _ in range(N)]
    for s in starts:
        s[:4] = rng.random(4).astype(np.float32)
    idx = [0]
    fig, (ax_traj, ax_blue, ax_red) = plt.subplots(1, 3, figsize=(18, 6.5))
    plt.subplots_adjust(bottom=0.18, wspace=0.3)
    status = fig.text(0.02, 0.01, "", fontsize=9)
    draw_policy_field(ax_blue, ax_red, env, net1, net2, device)
    fig.canvas.draw_idle()

    def render():
        traj, r1s, r2s = rollout(env, net1, net2, starts[idx[0]], T=T, device=device)
        s0 = starts[idx[0]]
        title = (f"{idx[0]+1}/{N}  p=({s0[0]:.2f},{s0[1]:.2f},{s0[2]:.2f},{s0[3]:.2f})  "
                 f"SR1={sum(r1s):.2f}  SR2={sum(r2s):.2f}")
        draw_traj(ax_traj, traj, env, title=title)
        status.set_text("Prev / Next to browse rollouts.")
        fig.canvas.draw_idle()

    ax_prev = fig.add_axes([0.44, 0.05, 0.08, 0.07])
    ax_next = fig.add_axes([0.54, 0.05, 0.08, 0.07])
    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")
    b_prev.on_clicked(lambda _: (idx.__setitem__(0, (idx[0]-1) % N), render()))
    b_next.on_clicked(lambda _: (idx.__setitem__(0, (idx[0]+1) % N), render()))
    render(); plt.show()


#  Entry point 

if __name__ == "__main__":
    t0 = time.time()

    env = DogGame(
        step=STEP, K_dirs=NUM_ACTIONS, add_stay=True, house_r=0.03,
        max_episode_steps=STEPS_PER_EPISODE, seed=0,
        houses_fixed=(0.25, 0.25, 0.75, 0.75),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net1, net2, loss1_history, loss2_history = train(env, device=device)

    print(f"Training time: {(time.time() - t0) / 60:.1f} minutes")
    plot_loss(loss1_history, loss2_history)
    browse(env, net1, net2, N=100, T=120, seed=0, device=device)