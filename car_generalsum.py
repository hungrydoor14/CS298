import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.9
TOL = 1e-3
MAX_Q_ITERS = 100
FP_ITERS = 100
CRASH_PENALTY = -10.0

# Actions: Up, Down, Left, Right
ACTIONS = ['U', 'D', 'L', 'R']
A = list(range(len(ACTIONS)))

def best_response(Q, opp_policy):
    """
    Q: payoff matrix for current player
    opp_policy: opponent mixed strategy
    """

    # expected payoff for each action against opponent policy
    values = Q @ opp_policy

    # choose action that maximizes expected payoff
    a = np.argmax(values)

    # return as a pure strategy (one-hot)
    pi = np.zeros(len(values))
    pi[a] = 1.0
    return pi

def pure_nash_equilibria(Q, tol=1e-9):
    """
    Returns list of pure NE (a1, a2)
    """
    n1, n2 = Q.shape
    nes = []

    for a1 in range(n1):
        for a2 in range(n2):
            v = Q[a1, a2]

            # Player 1 deviation
            p1_best = np.max(Q[:, a2])
            # Player 2 deviation (minimizer)
            p2_best = np.min(Q[a1, :])

            if abs(v - p1_best) < tol and abs(v - p2_best) < tol:
                nes.append((a1, a2))

    return nes

def pure_nash_equilibria_general(Q1, Q2, tol=1e-9):
    n1, n2 = Q1.shape
    nes = []

    for a1 in range(n1):
        for a2 in range(n2):
            if (Q1[a1, a2] >= np.max(Q1[:, a2]) - tol and
                Q2[a1, a2] >= np.max(Q2[a1, :]) - tol):
                nes.append((a1, a2))

    return nes

def solve_stage_game_general(Q1, Q2):
    pure_nes = pure_nash_equilibria_general(Q1, Q2)

    if pure_nes:
        a1, a2 = max(pure_nes, key=lambda p: Q1[p] + Q2[p])

        pi1 = np.zeros(Q1.shape[0]); pi1[a1] = 1.0
        pi2 = np.zeros(Q2.shape[1]); pi2[a2] = 1.0

        return (Q1[a1, a2], Q2[a1, a2]), pi1, pi2, "pure"

    # independent best responses (not zero-sum FP)
    pi1 = best_response(Q1, np.ones(Q2.shape[1]) / Q2.shape[1])
    pi2 = best_response(Q2.T, np.ones(Q1.shape[0]) / Q1.shape[0])

    v1 = pi1 @ Q1 @ pi2
    v2 = pi1 @ Q2 @ pi2

    return (v1, v2), pi1, pi2, "br"

def make_grid_reward(n, R_max=5.0, alpha=1.0):
    c = (n - 1) / 2
    grid = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            dist = abs(x - c) + abs(y - c)
            grid[x, y] = R_max - alpha * dist
    return grid

# Car / Grid Game Environment
class CarGame:
    """
    Two-player zero-sum grid game.
    State: (x1, y1, x2, y2)
    Player 1 tries to collide with Player 2.
    """

    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.states = self._all_states()

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
        s_next = self.transition(s, a1, a2)
        x1, y1, x2, y2 = s_next

        if (x1, y1) == (x2, y2):
            return CRASH_PENALTY     
        
        grid_r = self.grid_reward[x1, y1]

        # Small living cost to avoid stalling forever
        return grid_r - 1

class GeneralSumCarGame(CarGame):
    """
    General-sum variant:
    P1 wants to collide
    P2 wants to avoid collision
    """
    def __init__(self, grid_size=5):
        super().__init__(grid_size)
        self.grid_reward = make_grid_reward(grid_size, R_max=1.0, alpha=0.2)

    def reward(self, s, a1, a2):
        s_next = self.transition(s, a1, a2)
        x1, y1, x2, y2 = s_next

        collide = (x1, y1) == (x2, y2)

        # Manhattan distance between players
        dist = abs(x1 - x2) + abs(y1 - y2)

        if collide:
            r1 = 10.0
            r2 = -10.0
        else:
            # Dog wants to minimize distance
            r1 = -0.2 * dist + 0.1 * self.grid_reward[x1, y1]

            # Prey wants to maximize distance
            r2 = +0.2 * dist + 0.1 * self.grid_reward[x2, y2]

        return r1, r2


    
def check_ne(Q, pi1, pi2, value, tol=1e-3):
    """
    Check whether (pi1, pi2) is an approximate Nash equilibrium
    for zero-sum matrix game Q.
    AID FROM CHATGPT
    """

    # Player 1 unilateral deviations
    p1_deviation = False
    for a1 in range(Q.shape[0]):
        payoff = Q[a1] @ pi2
        if payoff > value + tol:
            p1_deviation = True

    # Player 2 unilateral deviations
    p2_deviation = False
    for a2 in range(Q.shape[1]):
        payoff = pi1 @ Q[:, a2]
        if payoff < value - tol:
            p2_deviation = True

    return not p1_deviation and not p2_deviation

def max_deviation(Q, pi1, pi2, value):
    """
    Maximum unilateral deviation (epsilon) from Nash equilibrium.
    AID FROM CHATGPT
    """
    max_dev = 0.0

    # Player 1 deviations
    for a1 in range(Q.shape[0]):
        max_dev = max(max_dev, Q[a1] @ pi2 - value)

    # Player 2 deviations
    for a2 in range(Q.shape[1]):
        max_dev = max(max_dev, value - pi1 @ Q[:, a2])

    return max_dev

def argmax_random_tie(p):
    """
    Select an action from a mixed strategy with random tie-breaking
    """
    m = np.max(p)
    return np.random.choice(np.where(np.abs(p - m) < 1e-9)[0])

# Markov Game Q-Iteration

def markov_game_q_iteration_general(env):
    Q1 = defaultdict(lambda: np.zeros((len(A), len(A))))
    Q2 = defaultdict(lambda: np.zeros((len(A), len(A))))

    V1 = defaultdict(float)
    V2 = defaultdict(float)

    Pi1, Pi2 = {}, {}

    for it in range(MAX_Q_ITERS):
        delta = 0.0

        for s in env.states:
            V1_old, V2_old = V1[s], V2[s]

            if (s[0], s[1]) == (s[2], s[3]):
                V1[s] = V2[s] = 0.0
                continue

            for a1 in A:
                for a2 in A:
                    r1, r2 = env.reward(s, a1, a2)
                    s_next = env.transition(s, a1, a2)

                    Q1[s][a1, a2] = r1 + GAMMA * V1[s_next]
                    Q2[s][a1, a2] = r2 + GAMMA * V2[s_next]

            (v1, v2), pi1, pi2, method = solve_stage_game_general(Q1[s], Q2[s])

            V1[s], V2[s] = v1, v2
            Pi1[s], Pi2[s] = pi1, pi2

            delta = max(delta,
                        abs(V1[s] - V1_old),
                        abs(V2[s] - V2_old))
        if it % 10 == 0:
            print(f"Gen-sum iter {it}, delta = {delta:.6f}")

        if delta < TOL:
            print("Converged (heuristically).")
            break

    return Q1, Q2, V1, V2, Pi1, Pi2


# Plotting utilities to make it graph NE actions
def deterministic_policy(Pi1, Pi2):
    policy = {}
    for s in Pi1:
        a1 = argmax_random_tie(Pi1[s])
        a2 = argmax_random_tie(Pi2[s])
        policy[s] = (a1, a2)
    return policy

def stochastic_policy(Pi1, Pi2):
    policy = {}
    for s in Pi1:
        pi1 = Pi1[s]
        pi2 = Pi2[s]
        policy[s] = (
            lambda pi=pi1: np.random.choice(A, p=pi),
            lambda pi=pi2: np.random.choice(A, p=pi)
        )
    return policy

def rollout(env, s0, policy, T=30):
    """
    Roll out an equilibrium trajectory from an initial state
    """
    traj = [s0]
    s = s0
    for _ in range(T):
        a1 = policy[s][0]()
        a2 = policy[s][1]()
        s = env.transition(s, a1, a2)
        traj.append(s)
        if (s[0], s[1]) == (s[2], s[3]):
            break
    return traj


def plot_trajectory(traj, grid_size, title=""):
    """
    Plot the trajectory of both players on the grid with arrows and colors.
    Added offset to distinguish lanes.
    1: Red (Player 1)
    2: Blue (Player 2)
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_title(title)

    offset = 0.12

    # Fixed lane offsets (constant per player)
    off1 = np.array([ offset, 0.0])
    off2 = np.array([-offset, 0.0])

    for s, s_next in zip(traj[:-1], traj[1:]):

        # Player 1
        p1_start = np.array([s[0] + 0.5, s[1] + 0.5]) + off1
        p1_end   = np.array([s_next[0] + 0.5, s_next[1] + 0.5]) + off1
        d1 = p1_end - p1_start

        ax.arrow(
            p1_start[0], p1_start[1],
            d1[0], d1[1],
            head_width=0.15,
            color="red",
            length_includes_head=True
        )

        # Player 2
        p2_start = np.array([s[2] + 0.5, s[3] + 0.5]) + off2
        p2_end   = np.array([s_next[2] + 0.5, s_next[3] + 0.5]) + off2
        d2 = p2_end - p2_start

        ax.arrow(
            p2_start[0], p2_start[1],
            d2[0], d2[1],
            head_width=0.15,
            color="blue",
            length_includes_head=True
        )

    plt.show()

# Run the car game
if __name__ == "__main__":
    env = GeneralSumCarGame(grid_size=5)
    Q1, Q2, V1, V2, Pi1_gs, Pi2_gs = markov_game_q_iteration_general(env)

    policy = stochastic_policy(Pi1_gs, Pi2_gs)

    s0 = (1, 0, 2, 2)
    traj = rollout(env, s0, policy)

    plot_trajectory(
        traj,
        env.grid_size,
        title=f"General-sum trajectory (grid={env.grid_size}) from {s0}"
    )
