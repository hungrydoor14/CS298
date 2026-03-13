import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import random
from collections import deque, OrderedDict
import nashpy as nash

import torch
import torch.nn as nn
import torch.optim as optim

import time

GAMMA = 0.65
STEPS = 100_000
EPS_DECAY = 50_000
LR = 1e-4
N_ITERS = 30

STEP = 0.05
NUM_ACTIONS = 8
STEPS_PER_EPISODE = 150
WARMUP = 5000
BATCH_SIZE = 64
CACHE_DECIMALS = 2

def best_response_p1(Q1, p2, mask1):
    v = Q1 @ p2
    v = np.where(mask1, v, -np.inf)
    i = int(np.argmax(v))
    p = np.zeros(len(mask1), dtype=np.float64)
    p[i] = 1.0
    return p

def best_response_p2(Q2, p1, mask2):
    v = p1 @ Q2
    v = np.where(mask2, v, -np.inf)
    j = int(np.argmax(v))
    p = np.zeros(len(mask2), dtype=np.float64)
    p[j] = 1.0
    return p

def solve_nash_gensum(Q1, Q2, mask1, mask2, n_iters=N_ITERS):
    p1 = mask1.astype(np.float64); p1 /= p1.sum()
    p2 = mask2.astype(np.float64); p2 /= p2.sum()
    for _ in range(n_iters):
        p1 = best_response_p1(Q1, p2, mask1)
        p2 = best_response_p2(Q2, p1, mask2)
    v1 = float(p1 @ Q1 @ p2)
    v2 = float(p1 @ Q2 @ p2)
    return v1, v2, p1, p2

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
        # Actions: K_dirs at full step, then K_dirs at half step, then optional stay
        # e.g. K_dirs=8 -> 16 directional actions + optional stay = 17 total
        self.K = self.K_dirs * 2 + (1 if self.add_stay else 0)
        self.stay_idx = self.K_dirs * 2 if self.add_stay else None
        base_angles = (2.0 * np.pi) * (np.arange(self.K_dirs) / self.K_dirs)
        self.angles = base_angles  # shared angles for both rings
        # step sizes per action: first K_dirs at full step, next K_dirs at half step
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

    def masks(self, s):
        m = np.ones(self.K, dtype=bool)
        return m.copy(), m.copy()

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


class QNet(nn.Module):
    def __init__(self, state_dim=8, K=6, hidden=128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, K * K),
        )

    def forward(self, s):
        return self.net(s).view(-1, self.K, self.K)


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


class LRUCache:
    def __init__(self, cap=50_000):
        self.cap = cap
        self.d = OrderedDict()

    def get(self, key):
        if key not in self.d:
            return None
        self.d.move_to_end(key)
        return self.d[key]

    def put(self, key, val):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = val
        if len(self.d) > self.cap:
            self.d.popitem(last=False)

    def clear(self):
        self.d.clear()

    def __len__(self):
        return len(self.d)


def _key(s, dec=1):
    return tuple(np.round(s[:4].astype(np.float64), dec))


@torch.no_grad()
def soft_update(tgt, src, tau=0.005):
    for pt, ps in zip(tgt.parameters(), src.parameters()):
        pt.data.mul_(1 - tau).add_(tau * ps.data)


@torch.no_grad()
def greedy_actions(net1, net2, env, s_np, device):
    s = torch.tensor(s_np[None], dtype=torch.float32, device=device)
    Q1 = net1(s)[0].cpu().numpy()
    Q2 = net2(s)[0].cpu().numpy()
    m1, m2 = env.masks(s_np)
    p2_unif = m2.astype(np.float64); p2_unif /= p2_unif.sum()
    p1_unif = m1.astype(np.float64); p1_unif /= p1_unif.sum()
    v1 = np.where(m1, Q1 @ p2_unif, -np.inf)
    v2 = np.where(m2, p1_unif @ Q2, -np.inf)
    return int(np.argmax(v1)), int(np.argmax(v2))


@torch.no_grad()
def nash_actions(net1, net2, env, s_np, device):
    s = torch.tensor(s_np[None], dtype=torch.float32, device=device)
    Q1 = net1(s)[0].cpu().numpy()
    Q2 = net2(s)[0].cpu().numpy()
    m1, m2 = env.masks(s_np)
    _, _, p1, p2 = solve_nash_gensum(Q1, Q2, m1, m2)
    return int(np.argmax(p1)), int(np.argmax(p2))


def train(env, steps=STEPS, warmup=WARMUP, batch_size=BATCH_SIZE, lr=LR,
          eps_start=1.0, eps_end=0.05, eps_decay_steps=EPS_DECAY,
          target_sync=100, grad_clip=5.0, hidden=256, seed=0,
          device=None, cache_decimals=CACHE_DECIMALS):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    K = env.K
    net1 = QNet(state_dim=8, K=K, hidden=hidden).to(device)
    net2 = QNet(state_dim=8, K=K, hidden=hidden).to(device)
    tgt1 = QNet(state_dim=8, K=K, hidden=hidden).to(device)
    tgt2 = QNet(state_dim=8, K=K, hidden=hidden).to(device)
    tgt1.load_state_dict(net1.state_dict()); tgt1.eval()
    tgt2.load_state_dict(net2.state_dict()); tgt2.eval()

    opt1 = optim.Adam(net1.parameters(), lr=lr)
    opt2 = optim.Adam(net2.parameters(), lr=lr)

    replay = Replay()
    cache = LRUCache(cap=80_000)
    rng = np.random.default_rng(seed + 1)

    s = env.reset()
    m1_all = np.ones(K, dtype=bool)
    m2_all = np.ones(K, dtype=bool)

    loss1_history = []
    loss2_history = []
    for t in range(1, steps + 1):
        frac = min(1.0, t / eps_decay_steps)
        eps = eps_start + frac * (eps_end - eps_start)

        if rng.random() < eps:
            a1 = int(rng.integers(K))
            a2 = int(rng.integers(K))
        else:
            a1, a2 = nash_actions(net1, net2, env, s, device)

        sn, r1, r2, done = env.step_env(s, a1, a2)
        replay.add(s, a1, a2, r1, r2, sn, done)
        s = env.reset() if done else sn

        if len(replay) < warmup:
            continue

        sb, a1b, a2b, r1b, r2b, snb, doneb = replay.sample(batch_size)

        sb_t   = torch.tensor(sb,   dtype=torch.float32, device=device)
        snb_t  = torch.tensor(snb,  dtype=torch.float32, device=device)
        r1b_t  = torch.tensor(r1b,  dtype=torch.float32, device=device)
        r2b_t  = torch.tensor(r2b,  dtype=torch.float32, device=device)
        done_t = torch.tensor(doneb, dtype=torch.float32, device=device)

        idx1 = torch.tensor(a1b, dtype=torch.int64, device=device).view(-1, 1, 1)
        idx2 = torch.tensor(a2b, dtype=torch.int64, device=device).view(-1, 1, 1)

        with torch.no_grad():
            Q1n = tgt1(snb_t).cpu().numpy()
            Q2n = tgt2(snb_t).cpu().numpy()

        v1_next = np.zeros(batch_size, dtype=np.float32)
        v2_next = np.zeros(batch_size, dtype=np.float32)
        miss_idx = {}

        for i in range(batch_size):
            key = _key(snb[i], cache_decimals)
            hit = cache.get(key)
            if hit is not None:
                v1_next[i], v2_next[i] = hit
            else:
                miss_idx.setdefault(key, []).append(i)

        for key, idxs in miss_idx.items():
            i0 = idxs[0]
            v1, v2, _, _ = solve_nash_gensum(Q1n[i0], Q2n[i0], m1_all, m2_all)
            cache.put(key, (v1, v2))
            for i in idxs:
                v1_next[i] = v1
                v2_next[i] = v2

        v1t = torch.tensor(v1_next, dtype=torch.float32, device=device)
        v2t = torch.tensor(v2_next, dtype=torch.float32, device=device)

        y1 = r1b_t + (1 - done_t) * GAMMA * v1t
        y2 = r2b_t + (1 - done_t) * GAMMA * v2t

        Q1_pred = net1(sb_t)
        Q1_sa = Q1_pred.gather(1, idx1.expand(-1, 1, K)).gather(2, idx2).squeeze(-1).squeeze(-1)
        loss1 = torch.nn.functional.smooth_l1_loss(Q1_sa, y1)
        opt1.zero_grad(); loss1.backward()
        torch.nn.utils.clip_grad_norm_(net1.parameters(), grad_clip)
        opt1.step()

        Q2_pred = net2(sb_t)
        Q2_sa = Q2_pred.gather(1, idx1.expand(-1, 1, K)).gather(2, idx2).squeeze(-1).squeeze(-1)
        loss2 = torch.nn.functional.smooth_l1_loss(Q2_sa, y2)
        opt2.zero_grad(); loss2.backward()
        torch.nn.utils.clip_grad_norm_(net2.parameters(), grad_clip)
        opt2.step()

        if t % target_sync == 0:
            soft_update(tgt1, net1)
            soft_update(tgt2, net2)

        if t % 2000 == 0:
            cache.clear()

        loss1_history.append(float(loss1))
        loss2_history.append(float(loss2))

        if t % 500 == 0:
            print(f"step {t:5d}  eps {eps:.3f}  "
                  f"loss1 {float(loss1):.5f}  loss2 {float(loss2):.5f}  "
                  f"replay {len(replay)}  cache {len(cache)}")

    return net1, net2, loss1_history, loss2_history


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
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.show()


def rollout(env, net1, net2, s0, T=100, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    s = env.reset(s0=s0)
    traj = [s.copy()]
    r1s, r2s = [], []
    for _ in range(T):
        a1, a2 = nash_actions(net1, net2, env, s, device)
        sn, r1, r2, done = env.step_env(s, a1, a2)
        traj.append(sn.copy()); r1s.append(r1); r2s.append(r2)
        s = sn
        if done:
            break
    return traj, r1s, r2s


def draw_traj(ax, traj, env, title=None):
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=9)

    h1x, h1y, h2x, h2y = map(float, traj[0][4:])
    ax.scatter([h1x], [h1y], marker="s", s=200, color="blue", zorder=5)
    ax.scatter([h2x], [h2y], marker="s", s=200, color="red",  zorder=5)

    arr = np.array(traj)
    p1  = arr[:, :2]
    p2  = arr[:, 2:4]
    dog = 0.5 * (p1 + p2)

    hw = 0.012
    for k in range(len(traj) - 1):
        dx, dy = p1[k+1] - p1[k]
        if abs(dx) + abs(dy) > 1e-6:
            ax.arrow(p1[k,0], p1[k,1], dx, dy, color="blue", head_width=hw,
                     length_includes_head=True, zorder=3)
        dx, dy = p2[k+1] - p2[k]
        if abs(dx) + abs(dy) > 1e-6:
            ax.arrow(p2[k,0], p2[k,1], dx, dy, color="red", head_width=hw,
                     length_includes_head=True, zorder=3)
        dx, dy = dog[k+1] - dog[k]
        if abs(dx) + abs(dy) > 1e-6:
            ax.arrow(dog[k,0], dog[k,1], dx, dy, color="black", head_width=hw,
                     length_includes_head=True, zorder=4)

    ax.scatter(*p1[0],  marker="o", s=60,  color="blue",  zorder=6)
    ax.scatter(*p2[0],  marker="o", s=60,  color="red",   zorder=6)
    ax.scatter(*dog[0], marker="x", s=100, color="black", zorder=6)


@torch.no_grad()
@torch.no_grad()
def draw_policy_field(ax_blue, ax_red, env, net1, net2, device, grid_n=15):
    """
    Sweep both players over the grid simultaneously (blue at (px,py), red at mirror (1-px,1-py)).
    This is in-distribution since training randomises both positions.
    Blue panel shows blue arrow at each grid point, red panel shows red arrow.
    """
    h1x, h1y, h2x, h2y = map(float, env.houses_fixed)
    K = env.K

    lin = np.linspace(0.05, 0.95, grid_n)
    xs, ys = np.meshgrid(lin, lin)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)

    m = np.ones(K, dtype=bool)
    u1s, v1s, u2s, v2s = [], [], [], []
    for px, py in pts:
        s = np.array([px, py, 1-px, 1-py, h1x, h1y, h2x, h2y], dtype=np.float32)
        s_t = torch.tensor(s[None], dtype=torch.float32, device=device)
        Q1 = net1(s_t)[0].cpu().numpy()
        Q2 = net2(s_t)[0].cpu().numpy()
        _, _, p1, p2 = solve_nash_gensum(Q1, Q2, m, m)
        a1 = int(np.argmax(p1))
        a2 = int(np.argmax(p2))

        if env.add_stay and a1 == env.stay_idx:
            u1s.append(0.0); v1s.append(0.0)
        else:
            theta = float(env.action_angles[a1])
            u1s.append(np.cos(theta)); v1s.append(np.sin(theta))

        if env.add_stay and a2 == env.stay_idx:
            u2s.append(0.0); v2s.append(0.0)
        else:
            theta = float(env.action_angles[a2])
            u2s.append(np.cos(theta)); v2s.append(np.sin(theta))

    scale = 0.9 / grid_n
    for ax, color, us, vs_arr, title in [
        (ax_blue, "blue", np.array(u1s), np.array(v1s), "Blue policy (red at mirror pos)"),
        (ax_red,  "red",  np.array(u2s), np.array(v2s), "Red policy (blue at mirror pos)"),
    ]:
        ax.clear()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
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
        title = (f"{idx[0]+1}/{N}  "
                 f"p=({s0[0]:.2f},{s0[1]:.2f},{s0[2]:.2f},{s0[3]:.2f})  "
                 f"SR1={sum(r1s):.2f}  SR2={sum(r2s):.2f}")
        draw_traj(ax_traj, traj, env, title=title)
        status.set_text("Prev / Next to browse rollouts. Policy fields are fixed.")
        fig.canvas.draw_idle()
        final = traj[-1]
        print(f"Final: blue=({final[0]:.3f},{final[1]:.3f}) red=({final[2]:.3f},{final[3]:.3f})")


    ax_prev = fig.add_axes([0.44, 0.05, 0.08, 0.07])
    ax_next = fig.add_axes([0.54, 0.05, 0.08, 0.07])

    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")

    b_prev.on_clicked(lambda _: (idx.__setitem__(0, (idx[0]-1) % N), render()))
    b_next.on_clicked(lambda _: (idx.__setitem__(0, (idx[0]+1) % N), render()))

    render()
    plt.show()

def export_weights_txt(net, filename):
    with open(filename, 'w') as f:
        layers = []
        current_weights = None
        for name, p in net.named_parameters():
            if 'weight' in name:
                current_weights = p.detach().cpu().numpy()  # (out, in)
            elif 'bias' in name:
                bias = p.detach().cpu().numpy()  # (out,)
                # each row = one output neuron: [w1, w2, ..., wn, bias]
                combined = np.hstack([current_weights, bias.reshape(-1, 1)])  # (out, in+1)
                layers.append(combined)
        for i, arr in enumerate(layers):
            for row in arr:
                f.write(','.join(f'{x:.6f}' for x in row) + '\n')
            if i < len(layers) - 1:
                f.write('-----\n')

def load_weights_txt(net, filename, device=None):
    """Load weights from file. Infers architecture from file shape — no metadata needed.
    Pass net=None to auto-build the correctly-sized QNet from the file alone.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
 
    sections = []
    current = []
    for line in lines:
        if '-----' in line:
            sections.append(current); current = []
        else:
            current.append(line.strip())
    sections.append(current)
 
    arrays = []
    for section in sections:
        rows = [list(map(float, l.split(','))) for l in section if l]
        arrays.append(np.array(rows))
 
    if net is None:
        # Infer from shapes: section 0 is (hidden, state_dim+1), last is (K*K, hidden+1)
        state_dim = arrays[0].shape[1] - 1
        hidden    = arrays[0].shape[0]
        import math
        K = int(round(math.sqrt(arrays[-1].shape[0])))
        net = QNet(state_dim=state_dim, K=K, hidden=hidden)
 
    if device is not None:
        net = net.to(device)
 
    params = list(net.named_parameters())
    i = 0
    for arr in arrays:
        weights = arr[:, :-1]
        bias    = arr[:, -1]
        while i < len(params) and 'weight' not in params[i][0]:
            i += 1
        if i < len(params):
            params[i][1].data = torch.tensor(weights, dtype=torch.float32).to(params[i][1].device)
            i += 1
        while i < len(params) and 'bias' not in params[i][0]:
            i += 1
        if i < len(params):
            params[i][1].data = torch.tensor(bias, dtype=torch.float32).to(params[i][1].device)
            i += 1
 
    return net

if __name__ == "__main__":
    time1 = time.time()

    env = DogGame(
        step=STEP,
        K_dirs=NUM_ACTIONS,
        add_stay=True,
        house_r=0.03,
        max_episode_steps=STEPS_PER_EPISODE,
        seed=0,
        houses_fixed=(0.25, 0.25, 0.75, 0.75),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train
    net1, net2, loss1_history, loss2_history = train(env)

    # After training, save the weights
    export_weights_txt(net1, 'player1_weights.txt')
    export_weights_txt(net2, 'player2_weights.txt')

    time2 = time.time()
    print(f"Training time: {(time2 - time1) / 60:.1f} minutes")

    plot_loss(loss1_history, loss2_history)
    browse(env, net1, net2, N=100, T=120, seed=0, device=device)