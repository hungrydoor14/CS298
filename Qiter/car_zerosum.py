import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Hyperparameters
GAMMA = 0.9
TOL = 1e-3
MAX_Q_ITERS = 200
FP_ITERS = 800

GRID_SIZE = 3

CRASH_PENALTY = -10.0
STAY_PENALTY = -5.0
LIVING_COST = 1.0

POLICY_REFRESH = 10      # re-solve NE every N iterations
FP_ITERS_Q = 100         # cheap FP inside Q-iteration

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

def fictitious_play(Q, iters):
    """
    Solves NE(Q) for zero-sum game using FP
    """

    # initialize mixed strategies uniformly / num of acts
    n = Q.shape[0]
    pi1 = np.ones(n) / n
    pi2 = np.ones(n) / n

    # get histograms of action count
    hist1 = np.zeros(n)
    hist2 = np.zeros(n)

    for t in range(1, iters + 1):
        # best response to opponent's current empirical strategy
        br1 = best_response(Q, pi2)
        br2 = best_response(-Q.T, pi1)

        # update counts
        hist1 += br1
        hist2 += br2

        # convert counts into mixed strategies
        pi1 = hist1 / t
        pi2 = hist2 / t

    # game val under strats
    value = pi1 @ Q @ pi2
    return value, pi1, pi2, "fict"

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

def solve_stage_game(Q):
    """
    Solve the stage game using a pure Nash equilibrium if one exists
    """
    pure_nes = pure_nash_equilibria(Q)

    if pure_nes:
        a1, a2 = max(pure_nes, key=lambda pair: Q[pair])
        n = Q.shape[0]

        pi1 = np.zeros(n); pi1[a1] = 1.0
        pi2 = np.zeros(n); pi2[a2] = 1.0
        value = Q[a1, a2]

        return value, pi1, pi2, "pure"
    
    return None

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

# Markov Game Q-Iteration
def markov_game_q_iteration(env):
    """
    Optimized Q-iteration for a zero-sum Markov game.

    Key tricks (from notes):
    - Solve NE(Q) occasionally
    - Cache policies (pi1, pi2)
    - Use V(s) = pi1^T Q(s) pi2 in between (fast)
    - Compare equilibrium value vs best responses
    """

    Q = defaultdict(lambda: np.zeros((len(A), len(A))))
    V = defaultdict(float)

    Pi1 = {}
    Pi2 = {}

    POLICY_REFRESH = 10      # re-solve NE every N iterations
    FP_ITERS_Q = 100         # cheap FP inside Q-iteration

    for it in range(MAX_Q_ITERS):
        delta = 0.0

        for s in env.states:
            Q_old = Q[s].copy()

            # terminal collision state
            if (s[0], s[1]) == (s[2], s[3]):
                Q[s][:] = 0.0
                V[s] = 0.0
                Pi1[s] = np.zeros(len(A))
                Pi2[s] = np.zeros(len(A))
                continue

            # Bellman update for Q(s)
            for a1 in A:
                for a2 in A:
                    r = env.reward(s, a1, a2)
                    s_next = env.transition(s, a1, a2)
                    Q[s][a1, a2] = r + GAMMA * V[s_next]

            # STAGE GAME SOLVE / FAST PATH 
            if it % POLICY_REFRESH == 0 or s not in Pi1:
                # solve NE(Q) explicitly
                solution = solve_stage_game(Q[s])

                if solution is not None:
                    V[s], pi1, pi2, method = solution
                else:
                    V[s], pi1, pi2, method = fictitious_play(Q[s], FP_ITERS_Q)

                # round policies to reduce policy space
                pi1 = round_policy(pi1)
                pi2 = round_policy(pi2)

                Pi1[s] = pi1
                Pi2[s] = pi2

            else:
                # FAST VALUE UPDATE
                pi1 = Pi1[s]
                pi2 = Pi2[s]
                V[s] = pi1 @ Q[s] @ pi2

            delta = max(delta, np.max(np.abs(Q[s] - Q_old)))

        # progress print
        if it % 10 == 0:
            print(f"Q-iter {it:3d} | delta = {delta:.6f}")

        # convergence
        if delta < TOL:
            print("Converged.")
            break

    return Q, V, Pi1, Pi2

def round_policy(pi, grid=(0.0, 1/3, 1/2, 2/3, 1.0)):
    pi = np.array(pi, dtype=float)
    rounded = np.zeros_like(pi)

    for i, p in enumerate(pi):
        rounded[i] = min(grid, key=lambda g: abs(g - p))

    if rounded.sum() > 0:
        rounded /= rounded.sum()
    else:
        rounded[:] = 1.0 / len(rounded)

    return rounded

def stochastic_policy(Pi1, Pi2, eps=1e-12):
    policy = {}
    for s in Pi1:
        pi1 = Pi1[s].copy()
        pi2 = Pi2[s].copy()

        pi1 = np.maximum(pi1, 0)
        pi2 = np.maximum(pi2, 0)

        if pi1.sum() < eps:
            pi1 = np.ones(len(A)) / len(A)
        else:
            pi1 /= pi1.sum()

        if pi2.sum() < eps:
            pi2 = np.ones(len(A)) / len(A)
        else:
            pi2 /= pi2.sum()

        policy[s] = (
            lambda pi=pi1: np.random.choice(A, p=pi),
            lambda pi=pi2: np.random.choice(A, p=pi)
        )
    return policy


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

def equilibrium_vs_br_stats(Q, Pi1, Pi2, states):
    stats = []

    for s in states:
        pi1 = Pi1[s]
        pi2 = Pi2[s]

        V_eq = pi1 @ Q[s] @ pi2

        # best responses
        BR1 = np.max(Q[s] @ pi2)          # player 1 maximizer
        BR2 = np.min(pi1 @ Q[s])          # player 2 minimizer

        gap_p1 = BR1 - V_eq
        gap_p2 = V_eq - BR2
        eps = max(gap_p1, gap_p2)

        stats.append((V_eq, BR1, BR2, gap_p1, gap_p2, eps))

    return stats

# Run the car game
if __name__ == "__main__":
    env = CarGame(grid_size=GRID_SIZE)
    Q, V, Pi1, Pi2 = markov_game_q_iteration(env)

    # Inspect bes/worst state
    best = max(V.items(), key=lambda x: x[1])
    worst = min(V.items(), key=lambda x: x[1])

    print("Best state:", best)
    print("Worst state:", worst)

    # NE DIAGNOSTICS
    ne_true = 0
    epsilons = []

    for s in env.states:
        # Check if there is a solution already, if not use FP
        solution = solve_stage_game(Q[s])
        if solution is not None:
            value, pi1, pi2, method = solution
        else:
            value, pi1, pi2, method = fictitious_play(Q[s], FP_ITERS)

        is_ne = check_ne(Q[s], pi1, pi2, value)
        eps = max_deviation(Q[s], pi1, pi2, value)

        if is_ne:
            ne_true += 1
        #print(f"Epsilon = {eps:.6f} | State {s} | Method: {method} | NE: {is_ne}")
        epsilons.append(eps)

    print("\nNE diagnostics:")
    print(f"NE states: {ne_true} / {len(env.states)}")
    print(f"Max epsilon: {max(epsilons):.6f}")
    print(f"Mean epsilon: {np.mean(epsilons):.6f}")

    states = env.states
    policy = stochastic_policy(Pi1, Pi2)

    trajectories = [
        rollout(env, s, policy, T=20)
        for s in states
    ]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    idx = [0]  # mutable closure (IMPORTANT)

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
            title=f"NE trajectory from {s}", # future proofing, might use it instead later
            subtitle=subtitle
        )

        fig.suptitle(
            f"NE trajectory from {s}",
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



    

    