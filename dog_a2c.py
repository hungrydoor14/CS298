import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time

GAMMA = 0.65
STEPS = 100_000
LR = 1e-4

STEP = 0.05
NUM_ACTIONS = 8
STEPS_PER_EPISODE = 150


# ── Environment (unchanged) ───────────────────────────────────────────────────

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


# ── Networks ──────────────────────────────────────────────────────────────────

class ActorNet(nn.Module):
    """Outputs a probability distribution over K actions."""
    def __init__(self, state_dim=8, K=17, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, s):
        return torch.softmax(self.net(s), dim=-1)   # (batch, K)


class CriticNet(nn.Module):
    """Outputs a scalar state-value estimate V(s)."""
    def __init__(self, state_dim=8, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)              # (batch,)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sample_action(actor, s_np, device):
    """Sample an action from the actor's distribution; also return log-prob."""
    s = torch.tensor(s_np[None], dtype=torch.float32, device=device)
    probs = actor(s)                                    # (1, K)
    dist  = torch.distributions.Categorical(probs)
    a     = dist.sample()                               # (1,)
    return int(a.item()), dist.log_prob(a).squeeze(0)   # scalar log-prob


@torch.no_grad()
def greedy_action(actor, s_np, device):
    """Greedy (mode) action for evaluation / rollout."""
    s = torch.tensor(s_np[None], dtype=torch.float32, device=device)
    return int(actor(s).argmax(dim=1).item())


# ── Training ──────────────────────────────────────────────────────────────────

def train(env, steps=STEPS, lr=LR,
          actor_coef=1.0, critic_coef=0.5, entropy_coef=0.01,
          grad_clip=5.0, hidden=256, seed=0, device=None):
    """
    Independent A2C for two agents.

    Each agent owns:
      - an actor  π(a | s; φ)
      - a critic  V(s; θ)

    At every environment step we compute a 1-step TD advantage and immediately
    update both networks — no replay buffer needed (on-policy).

    Loss per agent (Algorithm 14):
      actor  loss = -Adv · log π(a | s)         (+ optional entropy bonus)
      critic loss = (y - V(s))²
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    K = env.K

    # One actor + critic per agent
    actor1  = ActorNet(state_dim=8, K=K, hidden=hidden).to(device)
    critic1 = CriticNet(state_dim=8,    hidden=hidden).to(device)
    actor2  = ActorNet(state_dim=8, K=K, hidden=hidden).to(device)
    critic2 = CriticNet(state_dim=8,    hidden=hidden).to(device)

    # Joint optimisers (actor + critic parameters together per agent)
    opt1 = optim.Adam(list(actor1.parameters()) + list(critic1.parameters()), lr=lr)
    opt2 = optim.Adam(list(actor2.parameters()) + list(critic2.parameters()), lr=lr)

    s = env.reset()
    actor_loss1_history, actor_loss2_history = [], []

    for t in range(1, steps + 1):

        # ── Step 6: sample actions from actor distributions ──────────────────
        s_t = torch.tensor(s[None], dtype=torch.float32, device=device)

        probs1 = actor1(s_t)                        # (1, K)
        probs2 = actor2(s_t)
        dist1  = torch.distributions.Categorical(probs1)
        dist2  = torch.distributions.Categorical(probs2)
        a1_t   = dist1.sample()
        a2_t   = dist2.sample()
        logp1  = dist1.log_prob(a1_t).squeeze(0)   # scalar
        logp2  = dist2.log_prob(a2_t).squeeze(0)
        a1, a2 = int(a1_t.item()), int(a2_t.item())

        # ── Step 7: apply actions, observe r and s' ───────────────────────────
        sn, r1, r2, done = env.step_env(s, a1, a2)

        sn_t   = torch.tensor(sn[None], dtype=torch.float32, device=device)
        r1_t   = torch.tensor(r1, dtype=torch.float32, device=device)
        r2_t   = torch.tensor(r2, dtype=torch.float32, device=device)

        # ── Steps 8-13: compute TD targets and advantages ─────────────────────
        v1_s   = critic1(s_t).squeeze(0)
        v2_s   = critic2(s_t).squeeze(0)

        with torch.no_grad():
            v1_sn = critic1(sn_t).squeeze(0)
            v2_sn = critic2(sn_t).squeeze(0)

        if done:
            # Terminal: bootstrap value is 0  (Algorithm 14, lines 9-10)
            y1    = r1_t
            y2    = r2_t
            adv1  = r1_t - v1_s
            adv2  = r2_t - v2_s
        else:
            # Non-terminal: 1-step TD  (Algorithm 14, lines 12-13)
            y1    = r1_t + GAMMA * v1_sn
            y2    = r2_t + GAMMA * v2_sn
            adv1  = (r1_t + GAMMA * v1_sn) - v1_s
            adv2  = (r2_t + GAMMA * v2_sn) - v2_s

        # Detach advantages so actor gradient doesn't flow through critic
        adv1 = adv1.detach()
        adv2 = adv2.detach()

        # ── Steps 14-17: compute losses and update ────────────────────────────
        # Actor loss  = -Adv · log π(a|s)  (+ entropy regularisation bonus)
        # Critic loss = (y - V(s))²
        entropy1 = dist1.entropy().squeeze(0)
        entropy2 = dist2.entropy().squeeze(0)

        loss_actor1 = -adv1 * logp1 - entropy_coef * entropy1
        loss_critic1 = (y1.detach() - v1_s) ** 2
        loss1 = actor_coef * loss_actor1 + critic_coef * loss_critic1

        loss_actor2 = -adv2 * logp2 - entropy_coef * entropy2
        loss_critic2 = (y2.detach() - v2_s) ** 2
        loss2 = actor_coef * loss_actor2 + critic_coef * loss_critic2

        opt1.zero_grad(); loss1.backward()
        nn.utils.clip_grad_norm_(
            list(actor1.parameters()) + list(critic1.parameters()), grad_clip)
        opt1.step()

        opt2.zero_grad(); loss2.backward()
        nn.utils.clip_grad_norm_(
            list(actor2.parameters()) + list(critic2.parameters()), grad_clip)
        opt2.step()

        actor_loss1_history.append(float(loss_actor1))
        actor_loss2_history.append(float(loss_actor2))

        s = env.reset() if done else sn

        if t % 500 == 0:
            print(f"step {t:6d}  "
                  f"actor_loss1 {float(loss_actor1):+.5f}  "
                  f"actor_loss2 {float(loss_actor2):+.5f}  "
                  f"adv1 {float(adv1):+.4f}  adv2 {float(adv2):+.4f}")

    return actor1, actor2, actor_loss1_history, actor_loss2_history


# ── Visualisation ─────────────────────────────────────────────────────────────

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
    ax.set_xlabel("Training step"); ax.set_ylabel("Actor Loss")
    ax.set_title("Training Actor Loss"); ax.legend()
    plt.tight_layout(); plt.show()


def rollout(env, actor1, actor2, s0, T=100, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    s = env.reset(s0=s0)
    traj = [s.copy()]
    r1s, r2s = [], []
    for _ in range(T):
        a1 = greedy_action(actor1, s, device)
        a2 = greedy_action(actor2, s, device)
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
def draw_policy_field(ax_blue, ax_red, env, actor1, actor2, device, grid_n=15):
    h1x, h1y, h2x, h2y = map(float, env.houses_fixed)
    lin = np.linspace(0.05, 0.95, grid_n)
    xs, ys = np.meshgrid(lin, lin)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    u1s, v1s, u2s, v2s = [], [], [], []
    for px, py in pts:
        s = np.array([px, py, 1-px, 1-py, h1x, h1y, h2x, h2y], dtype=np.float32)
        a1 = greedy_action(actor1, s, device)
        a2 = greedy_action(actor2, s, device)
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


def browse(env, actor1, actor2, N=100, T=120, seed=0, device=None):
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
    draw_policy_field(ax_blue, ax_red, env, actor1, actor2, device)
    fig.canvas.draw_idle()

    def render():
        traj, r1s, r2s = rollout(env, actor1, actor2, starts[idx[0]], T=T, device=device)
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    env = DogGame(
        step=STEP, K_dirs=NUM_ACTIONS, add_stay=True, house_r=0.03,
        max_episode_steps=STEPS_PER_EPISODE, seed=0,
        houses_fixed=(0.25, 0.25, 0.75, 0.75),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor1, actor2, loss1_history, loss2_history = train(env, device=device)

    print(f"Training time: {(time.time() - t0) / 60:.1f} minutes")
    plot_loss(loss1_history, loss2_history)
    browse(env, actor1, actor2, N=100, T=120, seed=0, device=device)