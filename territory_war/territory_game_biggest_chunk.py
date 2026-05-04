from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Starting")

# Grid / game
GRID_SIZE = 30
TRAIN_GRID_SIZE = 10
TRAIN_MAX_STEPS = 100
ROLLOUT_MAX_STEPS = GRID_SIZE*GRID_SIZE + 50
START_GAP = 8
START_GAP_JITTER = 2
START_ROW_JITTER = 3

# Wall setup
WALLS_ENABLED = True
WALL = 3

# Training
LR = 3e-4
BATCH_SIZE = 64
GAMMA = 0.90
EPS_START = 0.35
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 50
UPDATE_EVERY = 4       # only update network every N steps
EPOCHS = 50
EPS_PER_EPOCH = 100

# timer interval
INTERVAL = 50

# Rewards
EMPTY_REWARD = 1.0
OWN_CELL_REWARD = -0.03
TERRITORY_ADVANTAGE_REWARD = 0.2
FRONTIER_PROGRESS_REWARD = 0.08
WIN_BONUS = 6.0
LOSE_PENALTY = -6.0
DRAW_BONUS = 1.0

SEED = 7

EMPTY = 0
RED = 1
BLUE = 2
PLAYER_NAMES = {RED: "Red", BLUE: "Blue"}

# Actions: 1=Up, 2=Right, 3=Down, 4=Left
ACTIONS = [1, 2, 3, 4]
ACTION_NAMES = {1: "Up", 2: "Right", 3: "Down", 4: "Left"}
A = [0, 1, 2, 3]  # indices into ACTIONS

# (drow, dcol) for each action
DELTAS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_state(env, player=None):
    n = env.size - 1
    r_pos = env.pos[RED]
    b_pos = env.pos[BLUE]
    player = env.active_player if player is None else player
    row, col = env.pos[player]
    enemy_row, enemy_col = env.pos[env._other_player(player)]
    frontier_distance = env.frontier_distance(player)
    biggest_empty_center = env.biggest_unclaimed_chunk_center()
    max_path = env.size * env.size
    frontier_feature = 1.0 if frontier_distance is None else frontier_distance / max_path
    if biggest_empty_center is None:
        chunk_row, chunk_col = row, col
        chunk_distance_feature = 1.0
    else:
        chunk_row, chunk_col = biggest_empty_center
        chunk_distance_feature = (abs(row - chunk_row) + abs(col - chunk_col)) / (2 * n)

    # Neighbor cell values for the active player: 0=empty, 1=own, -1=enemy, 2=wall
    neighbors = []
    for dr, dc in [(-1,0),(0,1),(1,0),(0,-1)]:
        nr, nc = row + dr, col + dc
        if not (0 <= nr < env.size and 0 <= nc < env.size):
            neighbors.append(2.0)   # wall
        else:
            cell = int(env.board[nr, nc])
            if cell == EMPTY:
                neighbors.append(0.0)
            elif cell == player:
                neighbors.append(1.0)
            elif cell == WALL:
                neighbors.append(2.0)
            else:
                neighbors.append(-1.0)

    return torch.tensor(
        [r_pos[0] / n, r_pos[1] / n, b_pos[0] / n, b_pos[1] / n]
        + neighbors
        + [
            (abs(row - enemy_row) + abs(col - enemy_col)) / (2 * n),
            frontier_feature,
            chunk_row / n,
            chunk_col / n,
            chunk_distance_feature,
        ],
        dtype=torch.float32, device=device
    )


class DQN(nn.Module):
    def __init__(self, state_dim=13, action_dim=4):
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


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, next_legal_mask, done):
        self.buffer.append((s, a, r, s_next, next_legal_mask, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.buffer = ReplayBuffer()
        self.loss_fn = nn.SmoothL1Loss()
        self.update_steps = 0
        self.transition_steps = 0

    def select_action(self, s_tensor, legal_actions, eps):
        if np.random.rand() < eps:
            return random.choice(legal_actions)
        with torch.no_grad():
            q = self.net(s_tensor)
        # mask illegal actions
        legal_idx = [a - 1 for a in legal_actions]
        best_idx = max(legal_idx, key=lambda i: q[i].item())
        return ACTIONS[best_idx]

    def greedy_action(self, s_tensor, legal_actions):
        return self.select_action(s_tensor, legal_actions, eps=0.0)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0.0

        batch = self.buffer.sample(BATCH_SIZE)
        s, a, r, sn, next_legal_mask, done = zip(*batch)

        s_batch  = torch.stack(s)
        sn_batch = torch.stack(sn)
        a_batch  = torch.tensor([ai - 1 for ai in a], device=device)
        r_batch  = torch.tensor(r, dtype=torch.float32, device=device)
        next_legal_mask_batch = torch.tensor(next_legal_mask, dtype=torch.bool, device=device)
        done_batch = torch.tensor(done, dtype=torch.float32, device=device)

        q_vals = self.net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_all = self.target_net(sn_batch)
            next_q_all = next_q_all.masked_fill(~next_legal_mask_batch, -1e9)
            next_q = next_q_all.max(1)[0]
            next_q = torch.where(next_q < -1e8, torch.zeros_like(next_q), next_q)
            target = r_batch + GAMMA * next_q * (1 - done_batch)

        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return loss.item()

    def should_update(self):
        return len(self.buffer) >= BATCH_SIZE and self.transition_steps % UPDATE_EVERY == 0


class TerritoryEnv:
    def __init__(self, size=GRID_SIZE, max_steps=TRAIN_MAX_STEPS):
        self.size = size
        self.max_steps = max_steps
        self.board = np.zeros((size, size), dtype=np.int8)
        self.pos = {}
        self.active_player = RED
        self.steps_taken = 0
        self.reset_count = 0
        self.reset()

    def reset(self):
        self.board.fill(EMPTY)

        if WALLS_ENABLED:
            wall_col = self.size // 2

            for r in range(self.size // 2, self.size):
                self.board[r, wall_col] = WALL

        center_row = self.size // 2 + random.randint(-START_ROW_JITTER, START_ROW_JITTER)
        center_row = max(0, min(self.size - 1, center_row))
        gap = START_GAP + random.randint(-START_GAP_JITTER, START_GAP_JITTER)
        gap = max(2, min(self.size - 2, gap))
        half_gap = max(1, gap // 2)
        center_col = self.size // 2
        left_start = (center_row, max(0, center_col - half_gap))
        right_start = (center_row, min(self.size - 1, center_col + half_gap))
        if left_start == right_start:
            right_start = (center_row, min(self.size - 1, left_start[1] + 1))

        # Randomize side assignment and first player so agents cannot memorize one mirrored opening.
        if random.random() < 0.5:
            red_start, blue_start = left_start, right_start
        else:
            red_start, blue_start = right_start, left_start
        self.active_player = RED if random.random() < 0.5 else BLUE

        self.pos = {RED: red_start, BLUE: blue_start}
        self.board[red_start] = RED
        self.board[blue_start] = BLUE
        self.steps_taken = 0
        self.reset_count += 1
        return self.board.copy()

    def _other_player(self, player=None):
        current = self.active_player if player is None else player
        return BLUE if current == RED else RED

    def frontier_distance_from(self, start, player=None):
        current = self.active_player if player is None else player
        enemy = self._other_player(current)
        queue = deque([(start[0], start[1], 0)])
        visited = {start}

        while queue:
            row, col, dist = queue.popleft()
            for dr, dc in DELTAS.values():
                nr, nc = row + dr, col + dc
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    continue
                if (nr, nc) in visited:
                    continue
                cell = int(self.board[nr, nc])
                if cell == WALL:
                    continue
                if cell == enemy:
                    continue
                if cell == EMPTY:
                    return dist + 1
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

        return None

    def legal_actions(self, player=None):
        current = self.active_player if player is None else player
        row, col = self.pos[current]
        empty_actions = []
        own_actions = []
        for a in ACTIONS:
            dr, dc = DELTAS[a]
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                cell = int(self.board[nr, nc])
                if cell == WALL:
                    continue
                elif cell == EMPTY:
                    empty_actions.append(a)
                elif cell == current:
                    own_actions.append(a)

        # Always prefer immediate expansion over wandering on owned cells.
        if empty_actions:
            return empty_actions

        if not own_actions:
            return []

        # If boxed into owned territory, only allow moves that stay on a shortest route
        # to the nearest reachable empty cell.
        best_distance = None
        best_actions = []
        for action in own_actions:
            dr, dc = DELTAS[action]
            next_pos = (row + dr, col + dc)
            distance = self.frontier_distance_from(next_pos, current)
            if distance is None:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_actions = [action]
            elif distance == best_distance:
                best_actions.append(action)

        return best_actions

    def counts(self):
        red = int(np.sum(self.board == RED))
        blue = int(np.sum(self.board == BLUE))
        empty = int(np.sum(self.board == EMPTY))
        return {RED: red, BLUE: blue, EMPTY: empty}

    def biggest_unclaimed_chunk_center(self):
        visited = np.zeros((self.size, self.size), dtype=bool)
        best_chunk = []

        for start_row in range(self.size):
            for start_col in range(self.size):
                if visited[start_row, start_col] or int(self.board[start_row, start_col]) != EMPTY:
                    continue

                chunk = []
                queue = deque([(start_row, start_col)])
                visited[start_row, start_col] = True

                while queue:
                    row, col = queue.popleft()
                    chunk.append((row, col))

                    for dr, dc in DELTAS.values():
                        nr, nc = row + dr, col + dc
                        if not (0 <= nr < self.size and 0 <= nc < self.size):
                            continue
                        if visited[nr, nc] or int(self.board[nr, nc]) != EMPTY:
                            continue

                        visited[nr, nc] = True
                        queue.append((nr, nc))

                if len(chunk) > len(best_chunk):
                    best_chunk = chunk

        if not best_chunk:
            return None

        center_row = sum(row for row, _ in best_chunk) / len(best_chunk)
        center_col = sum(col for _, col in best_chunk) / len(best_chunk)
        return min(
            best_chunk,
            key=lambda cell: (
                (cell[0] - center_row) ** 2 + (cell[1] - center_col) ** 2,
                cell[0],
                cell[1],
            ),
        )

    def frontier_distance(self, player=None):
        current = self.active_player if player is None else player
        return self.frontier_distance_from(self.pos[current], current)

    def can_reach_empty(self, player=None):
        return self.frontier_distance(player) is not None

    def winner(self):
        counts = self.counts()
        if counts[RED] > counts[BLUE]:
            return RED
        if counts[BLUE] > counts[RED]:
            return BLUE
        return None

    def step(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action} for player {self.active_player}")

        player = self.active_player
        enemy = self._other_player()
        prev_frontier_distance = self.frontier_distance(player)
        row, col = self.pos[player]
        dr, dc = DELTAS[action]
        nr, nc = row + dr, col + dc

        previous_owner = int(self.board[nr, nc])
        reward = EMPTY_REWARD if previous_owner == EMPTY else OWN_CELL_REWARD

        self.board[nr, nc] = player
        self.pos[player] = (nr, nc)

        new_frontier_distance = self.frontier_distance(player)
        if prev_frontier_distance is not None and new_frontier_distance is not None:
            reward += FRONTIER_PROGRESS_REWARD * (prev_frontier_distance - new_frontier_distance)
        elif prev_frontier_distance is not None and new_frontier_distance is None:
            reward -= FRONTIER_PROGRESS_REWARD

        counts = self.counts()
        territory_advantage = counts[player] - counts[enemy]
        reward += TERRITORY_ADVANTAGE_REWARD * (territory_advantage / (self.size * self.size))

        self.steps_taken += 1

        next_player = self._other_player()
        done = (
            self.steps_taken >= self.max_steps
            or (not self.can_reach_empty(RED) and not self.can_reach_empty(BLUE))
            or len(self.legal_actions(next_player)) == 0
        )
        info = {
            "row": nr,
            "col": nc,
            "action": action,
            "claimed_from": previous_owner,
            "player": self.active_player,
            "step": self.steps_taken,
        }

        if not done:
            self.active_player = next_player

        return self.board.copy(), reward, done, info


def terminal_shaping(winner, player):
    if winner == player:
        return WIN_BONUS
    if winner is None:
        return DRAW_BONUS
    return LOSE_PENALTY


def legal_action_mask(legal_actions):
    mask = [False] * len(ACTIONS)
    for action in legal_actions:
        mask[action - 1] = True
    return mask


def with_max_steps(env, max_steps, fn):
    original = env.max_steps
    env.max_steps = max_steps
    try:
        return fn()
    finally:
        env.max_steps = original


def evaluate_policy(env, agents, episodes=50, max_steps=None):
    eval_max_steps = env.max_steps if max_steps is None else max_steps

    def _run():
        red_wins = 0
        blue_wins = 0
        draws = 0
        red_cells = []
        blue_cells = []
        for _ in range(episodes):
            env.reset()
            done = False
            while not done:
                s = encode_state(env)
                player = env.active_player
                action = agents[player].greedy_action(s, env.legal_actions(player))
                _, _, done, _ = env.step(action)
            winner = env.winner()
            counts = env.counts()
            red_cells.append(counts[RED])
            blue_cells.append(counts[BLUE])
            if winner == RED:
                red_wins += 1
            elif winner == BLUE:
                blue_wins += 1
            else:
                draws += 1
        return {
            "red_win_rate": red_wins / episodes,
            "blue_win_rate": blue_wins / episodes,
            "draw_rate": draws / episodes,
            "avg_red_cells": float(np.mean(red_cells)),
            "avg_blue_cells": float(np.mean(blue_cells)),
        }

    return with_max_steps(env, eval_max_steps, _run)


def run_episode(env, agents, eps):
    env.reset()
    ep_losses = []
    pending = {RED: None, BLUE: None}

    done = False
    while not done:
        player = env.active_player
        agent = agents[player]
        s = encode_state(env, player)

        # Complete this player's previous transition now that it is their turn again.
        if pending[player] is not None:
            prev_s, prev_action, prev_reward = pending[player]
            current_legal = env.legal_actions(player)
            agent.buffer.push(
                prev_s,
                prev_action,
                prev_reward,
                s,
                legal_action_mask(current_legal),
                False,
            )
            agent.transition_steps += 1
            pending[player] = None
            if agent.should_update():
                loss = agent.update()
                if loss > 0:
                    ep_losses.append(loss)

        legal = env.legal_actions(player)
        action = agent.select_action(s, legal, eps)

        _, reward, done, _ = env.step(action)
        pending[player] = (s, action, reward)

        if done:
            winner = env.winner()

            # Finalize the acting player's last move with terminal reward.
            final_state = encode_state(env, player)
            terminal_legal = legal_action_mask([])
            prev_s, prev_action, prev_reward = pending[player]
            agent.buffer.push(
                prev_s,
                prev_action,
                prev_reward + terminal_shaping(winner, player),
                final_state,
                terminal_legal,
                True,
            )
            agent.transition_steps += 1
            pending[player] = None
            if agent.should_update():
                loss = agent.update()
                if loss > 0:
                    ep_losses.append(loss)

            # Also finalize the opponent's pending move so blue does not get all terminal credit.
            other = env._other_player(player)
            other_pending = pending[other]
            if other_pending is not None:
                other_agent = agents[other]
                prev_s, prev_action, prev_reward = other_pending
                other_final_state = encode_state(env, other)
                other_agent.buffer.push(
                    prev_s,
                    prev_action,
                    prev_reward + terminal_shaping(winner, other),
                    other_final_state,
                    terminal_legal,
                    True,
                )
                other_agent.transition_steps += 1
                pending[other] = None
                if other_agent.should_update():
                    loss = other_agent.update()
                    if loss > 0:
                        ep_losses.append(loss)

    return float(np.mean(ep_losses)) if ep_losses else 0.0


def train(epochs=EPOCHS, eps_per_epoch=EPS_PER_EPOCH):
    set_seed(SEED)
    train_env = TerritoryEnv(size=TRAIN_GRID_SIZE, max_steps=TRAIN_MAX_STEPS)
    agents = {RED: DQNAgent(RED), BLUE: DQNAgent(BLUE)}
    losses = []

    for epoch in range(epochs):
        epoch_losses = []
        for ep in range(eps_per_epoch):
            global_ep = epoch * eps_per_epoch + ep
            EPS = EPS_END + (EPS_START - EPS_END) * np.exp(-global_ep / EPS_DECAY)
            ep_loss = run_episode(train_env, agents, EPS)
            epoch_losses.append(ep_loss)

        avg_loss = float(np.mean(epoch_losses))
        losses.append(avg_loss)        

        metrics = evaluate_policy(train_env, agents, episodes=20, max_steps=TRAIN_MAX_STEPS)
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"eps = {EPS:.3f}, "
            f"avg loss = {avg_loss:.4f}, "
            f"red WR = {metrics['red_win_rate']:.2f}, "
            f"blue win rate = {metrics['blue_win_rate']:.2f}, "
            f"draw rate = {metrics['draw_rate']:.2f}, "
            f"red cells = {metrics['avg_red_cells']:.1f}, "
            f"blue cells = {metrics['avg_blue_cells']:.1f}"
        )

    return train_env, agents, losses


def rollout_match(env, agents, max_steps=None):
    rollout_max_steps = env.max_steps if max_steps is None else max_steps

    def _run():
        env.reset()
        boards = [env.board.copy()]
        positions = [(env.pos[RED], env.pos[BLUE])]
        unclaimed_centers = [env.biggest_unclaimed_chunk_center()]
        moves = []
        done = False

        while not done:
            s = encode_state(env)
            player = env.active_player
            action = agents[player].greedy_action(s, env.legal_actions(player))
            board, reward, done, info = env.step(action)
            moves.append({
                "step": info["step"],
                "player": info["player"],
                "row": info["row"],
                "col": info["col"],
                "action": info["action"],
                "claimed_from": info["claimed_from"],
                "reward": reward,
            })
            boards.append(board.copy())
            positions.append((env.pos[RED], env.pos[BLUE]))
            unclaimed_centers.append(env.biggest_unclaimed_chunk_center())

        return boards, moves, positions, unclaimed_centers

    return with_max_steps(env, rollout_max_steps, _run)


def board_summary(env):
    counts = env.counts()
    winner = env.winner()
    winner_text = "Draw" if winner is None else PLAYER_NAMES[winner]
    return (
        f"Winner: {winner_text} | "
        f"Red cells: {counts[RED]} | "
        f"Blue cells: {counts[BLUE]} | "
        f"Empty: {counts[EMPTY]}"
    )


CMAP = ListedColormap([
    "#f5efe4",  # empty
    "#d1495b",  # red
    "#2d6cdf",  # blue
    "#222222"   # wall
])

def prerender_frames(boards):
    norm = plt.Normalize(vmin=0, vmax=3)
    return [CMAP(norm(b)) for b in boards]


def draw_board(im, frame, ax, move, subtitle, red_pos, blue_pos, unclaimed_center):
    im.set_data(frame)
    for line in ax.lines[:]:
        line.remove()

    ax.plot(red_pos[1], red_pos[0], "o", color="white", markersize=8,
            markeredgecolor="#8b0000", markeredgewidth=2)
    ax.plot(blue_pos[1], blue_pos[0], "o", color="white", markersize=8,
            markeredgecolor="#00008b", markeredgewidth=2)
    if unclaimed_center is not None:
        ax.plot(unclaimed_center[1], unclaimed_center[0], "x", color="#f4d35e",
                markersize=10, markeredgewidth=2.5)

    if move is None:
        ax.set_title("Initial board", fontsize=12, pad=10)
    else:
        claimed_from = int(move["claimed_from"])
        source = "empty" if claimed_from == EMPTY else PLAYER_NAMES[claimed_from]
        direction = ACTION_NAMES[int(move["action"])]
        ax.set_title(
            f"Step {int(move['step'])}: {PLAYER_NAMES[int(move['player'])]} moved {direction} "
            f"-> ({int(move['row'])}, {int(move['col'])}) from {source}",
            fontsize=12, pad=10,
        )

    ax.set_xlabel(subtitle, fontsize=11, labelpad=12)


if __name__ == "__main__":
    # TRAIN
    train_env, agents, losses = train()

    rollout_env = TerritoryEnv(size=GRID_SIZE, max_steps=ROLLOUT_MAX_STEPS)
    metrics = evaluate_policy(rollout_env, agents, episodes=100, max_steps=ROLLOUT_MAX_STEPS)

    print(
        f"Final eval ({ROLLOUT_MAX_STEPS} steps) | red win rate: {metrics['red_win_rate']:.3f}, "
        f"blue win rate: {metrics['blue_win_rate']:.3f}, "
        f"draw rate: {metrics['draw_rate']:.3f}, "
        f"red cells: {metrics['avg_red_cells']:.1f}, "
        f"blue cells: {metrics['avg_blue_cells']:.1f}"
    )

    # ROLLOUT
    boards, moves, positions, unclaimed_centers = rollout_match(rollout_env, agents, max_steps=ROLLOUT_MAX_STEPS)
    print(board_summary(rollout_env))

    # BUFFER FRAMES
    print("Buffering frames...")
    frames = prerender_frames(boards)

    # INTERACTIVE VIEWER
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.18)

    im = ax.imshow(frames[0], interpolation="nearest", aspect="equal")
    ax.set_xticks(np.arange(-0.5, boards[0].shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, boards[0].shape[0], 1), minor=True)
    ax.grid(which="minor", color="#202020", linestyle="-", linewidth=0.4)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    idx = [0]

    def update():
        move = None if idx[0] == 0 else moves[idx[0] - 1]
        subtitle = board_summary(rollout_env) if idx[0] == len(boards) - 1 else f"Step {idx[0]} of {len(boards) - 1}"
        red_pos, blue_pos = positions[idx[0]]
        draw_board(im, frames[idx[0]], ax, move, subtitle, red_pos, blue_pos, unclaimed_centers[idx[0]])
        fig.suptitle("Territory Self-Play Rollout", fontsize=16, y=0.97)
        fig.canvas.draw_idle()

    def next_state(event):
        idx[0] = (idx[0] + 1) % len(boards)
        update()

    def prev_state(event):
        idx[0] = (idx[0] - 1) % len(boards)
        update()

    playing = [False]
    timer = [None]

    def auto_step():
        if playing[0]:
            if idx[0] >= len(boards) - 1:
                playing[0] = False
                bplay.label.set_text("Play")
                fig.canvas.draw_idle()
                return

            idx[0] += 1
            update()
            if idx[0] >= len(boards) - 1:
                playing[0] = False
                bplay.label.set_text("Play")
                fig.canvas.draw_idle()
                return

            timer[0] = fig.canvas.new_timer(interval=INTERVAL)
            timer[0].add_callback(auto_step)
            timer[0].single_shot = True
            timer[0].start()

    def toggle_play(event):
        playing[0] = not playing[0]
        bplay.label.set_text("Pause" if playing[0] else "Play")
        if playing[0]:
            auto_step()
        elif timer[0] is not None:
            timer[0].stop()

    axprev = plt.axes([0.10, 0.05, 0.12, 0.075])
    axplay = plt.axes([0.38, 0.05, 0.12, 0.075])
    axnext = plt.axes([0.66, 0.05, 0.12, 0.075])

    bprev = Button(axprev, "Prev")
    bplay = Button(axplay, "Play")
    bnext = Button(axnext, "Next")

    bprev.on_clicked(prev_state)
    bplay.on_clicked(toggle_play)
    bnext.on_clicked(next_state)

    update()
    plt.show()

    # LOSS CURVE
    plt.figure()
    plt.plot(losses)
    plt.title("Territory DQN training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
