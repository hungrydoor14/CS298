from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button

# Grid / game
GRID_SIZE = 6
MAX_STEPS = 100

# Training
EPISODES = 60000
ALPHA = 0.2
GAMMA = 0.95
EPS_START = 0.35
EPS_END = 0.05
EPS_DECAY = 1800.0

# Rewards
EMPTY_REWARD = 1.0
CAPTURE_REWARD = 2.0
OWN_TERRITORY_PENALTY = -5.0
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

# (drow, dcol) for each action
DELTAS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class TerritoryEnv:
    def __init__(self, size=GRID_SIZE, max_steps=MAX_STEPS):
        self.size = size
        self.max_steps = max_steps
        self.board = np.zeros((size, size), dtype=np.int8)
        self.pos = {RED: (0, 0), BLUE: (size - 1, size - 1)}  # starting corners
        self.active_player = RED
        self.steps_taken = 0
        self.reset()

    def reset(self):
        self.board.fill(EMPTY)
        self.pos = {RED: (0, 0), BLUE: (self.size - 1, self.size - 1)}
        # Claim starting squares
        self.board[0, 0] = RED
        self.board[self.size - 1, self.size - 1] = BLUE
        self.active_player = RED
        self.steps_taken = 0
        return self.board.copy()

    def _other_player(self, player=None):
        current = self.active_player if player is None else player
        return BLUE if current == RED else RED

    def encode_state(self, player=None):
        current = self.active_player if player is None else player
        flat_board = tuple(int(x) for x in self.board.flatten())
        r_pos = self.pos[RED]
        b_pos = self.pos[BLUE]
        return flat_board + (r_pos[0], r_pos[1], b_pos[0], b_pos[1], current, self.steps_taken)

    def legal_actions(self, player=None):
        current = self.active_player if player is None else player
        row, col = self.pos[current]
        actions = []
        for a in ACTIONS:
            dr, dc = DELTAS[a]
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                # Can't move onto a cell already owned by yourself
                # if int(self.board[nr, nc]) != current:
                actions.append(a)
        return actions

    def counts(self):
        red = int(np.sum(self.board == RED))
        blue = int(np.sum(self.board == BLUE))
        empty = self.size * self.size - red - blue
        return {RED: red, BLUE: blue, EMPTY: empty}

    def winner(self):
        counts = self.counts()
        if counts[RED] > counts[BLUE]:
            return RED
        if counts[BLUE] > counts[RED]:
            return BLUE
        return None

    def step(self, action):
        row, col = self.pos[self.active_player]
        dr, dc = DELTAS[action]
        nr, nc = row + dr, col + dc

        previous_owner = int(self.board[nr, nc])
        if previous_owner == self.active_player:
            reward = OWN_TERRITORY_PENALTY
        elif previous_owner == EMPTY:
            reward = EMPTY_REWARD
        else:
            reward = CAPTURE_REWARD
        self.board[nr, nc] = self.active_player
        self.pos[self.active_player] = (nr, nc)
        self.steps_taken += 1

        next_player = self._other_player()
        done = self.steps_taken >= self.max_steps or len(self.legal_actions(next_player)) == 0
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


class QAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.q_table = defaultdict(lambda: defaultdict(float))

    def select_action(self, state, legal_actions, eps):
        if random.random() < eps:
            return random.choice(legal_actions)

        q_values = self.q_table[state]
        best_value = max(q_values[a] for a in legal_actions)
        best_actions = [a for a in legal_actions if q_values[a] == best_value]
        return random.choice(best_actions)

    def greedy_action(self, state, legal_actions):
        return self.select_action(state, legal_actions, eps=0.0)

    def update(self, state, action, reward, next_state, next_legal_actions, done):
        current_q = self.q_table[state][action]
        if done or not next_legal_actions:
            target = reward
        else:
            next_q = max(self.q_table[next_state][a] for a in next_legal_actions)
            target = reward + GAMMA * next_q
        updated = current_q + ALPHA * (target - current_q)
        self.q_table[state][action] = updated
        return abs(updated - current_q)


def terminal_shaping(winner, player):
    if winner == player:
        return WIN_BONUS
    if winner is None:
        return DRAW_BONUS
    return LOSE_PENALTY


def evaluate_policy(env, agents, episodes=50):
    red_wins = 0

    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            player = env.active_player
            state = env.encode_state(player)
            action = agents[player].greedy_action(state, env.legal_actions(player))
            _, _, done, _ = env.step(action)

        winner = env.winner()
        if winner == RED:
            red_wins += 1

    return red_wins / episodes


def train(episodes=EPISODES):
    set_seed(SEED)
    env = TerritoryEnv()
    agents = {RED: QAgent(RED), BLUE: QAgent(BLUE)}
    losses = []

    for ep in range(episodes):
        EPS = EPS_END + (EPS_START - EPS_END) * np.exp(-ep / EPS_DECAY)
        env.reset()
        done = False
        ep_updates = []

        while not done:
            player = env.active_player
            agent = agents[player]
            state = env.encode_state(player)
            legal = env.legal_actions(player)
            action = agent.select_action(state, legal, EPS)
            _, reward, done, _ = env.step(action)

            if done:
                reward += terminal_shaping(env.winner(), player)
                next_state = env.encode_state(player)
                next_legal = []
            else:
                next_state = env.encode_state(player)
                next_legal = env.legal_actions(player)

            delta = agent.update(state, action, reward, next_state, next_legal, done)
            ep_updates.append(delta)

        losses.append(float(np.mean(ep_updates)) if ep_updates else 0.0)

        if ep % 100 == 0:
            red_win_rate = evaluate_policy(env, agents, episodes=20)
            print(
                f"Episode {ep}, avg update = {losses[-1]:.4f}, "
                f"red win rate = {red_win_rate:.2f}"
            )

    return env, agents, losses


def rollout_match(env, agents):
    env.reset()
    boards = [env.board.copy()]
    positions = [(env.pos[RED], env.pos[BLUE])]
    moves = []
    done = False

    while not done:
        player = env.active_player
        state = env.encode_state(player)
        action = agents[player].greedy_action(state, env.legal_actions(player))
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

    return boards, moves, positions


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


def draw_board(ax, board, move, subtitle, red_pos, blue_pos):
    cmap = ListedColormap(["#f5efe4", "#d1495b", "#2d6cdf"])
    ax.clear()
    ax.imshow(board, cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(np.arange(-0.5, board.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, board.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#202020", linestyle="-", linewidth=1.2)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            value = int(board[row, col])
            label = "." if value == EMPTY else ("R" if value == RED else "B")
            color = "#1a1a1a" if value == EMPTY else "white"
            ax.text(col, row, label, ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    # Draw player position markers
    ax.plot(red_pos[1], red_pos[0], "o", color="white", markersize=14, markeredgecolor="#8b0000", markeredgewidth=2)
    ax.plot(blue_pos[1], blue_pos[0], "o", color="white", markersize=14, markeredgecolor="#00008b", markeredgewidth=2)

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
    env, agents, losses = train()

    red_win_rate = evaluate_policy(env, agents, episodes=100)
    print(f"Final red win rate: {red_win_rate:.3f}")

    # ROLLOUT
    boards, moves, positions = rollout_match(env, agents)
    print(board_summary(env))

    # INTERACTIVE VIEWER
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.18)

    idx = [0]

    def update():
        move = None if idx[0] == 0 else moves[idx[0] - 1]
        subtitle = board_summary(env) if idx[0] == len(boards) - 1 else f"Step {idx[0]} of {len(boards) - 1}"
        red_pos, blue_pos = positions[idx[0]]
        draw_board(ax, boards[idx[0]], move, subtitle, red_pos, blue_pos)
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
            idx[0] = (idx[0] + 1) % len(boards)
            update()
            timer[0] = fig.canvas.new_timer(interval=120)
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
    plt.title("Territory training loss")
    plt.xlabel("Episode")
    plt.ylabel("Average Q update")
    plt.show()