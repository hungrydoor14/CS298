import numpy as np
from collections import defaultdict

# Hyperparameters
GAMMA = 0.9
TOL = 1e-3
MAX_Q_ITERS = 200
FP_ITERS = 200

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
    return value, pi1, pi2

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

def solve_stage_game(Q, tol=1e-3):
    """
    1) Check pure NE
    2) If none, return None (meaning: run FP)
    """

    pure_nes = pure_nash_equilibria(Q)

    if pure_nes:
        a1, a2 = pure_nes[0]
        n = Q.shape[0]

        pi1 = np.zeros(n); pi1[a1] = 1.0
        pi2 = np.zeros(n); pi2[a2] = 1.0
        value = Q[a1, a2]

        return value, pi1, pi2

    return None

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
        """
        +1 if player 1 collides with player 2
        """
        s_next = self.transition(s, a1, a2)
        x1, y1, x2, y2 = s_next
        return 1.0 if (x1, y1) == (x2, y2) else 0.0
    
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
    Q-iteration for a zero-sum Markov game.
    Tries to solve each stage game exactly first (pure NE),
    falls back to fictitious play if needed.
    """

    Q = defaultdict(lambda: np.zeros((len(A), len(A))))
    V = defaultdict(float)

    for it in range(MAX_Q_ITERS):
        delta = 0.0

        for s in env.states:
            Q_old = Q[s].copy()

            # Update stage-game Q(s)
            for a1 in A:
                for a2 in A:
                    r = env.reward(s, a1, a2)
                    s_next = env.transition(s, a1, a2)
                    Q[s][a1, a2] = r + GAMMA * V[s_next]

            # Solve the stage game
            solution = solve_stage_game(Q[s])

            if solution is not None:
                V[s], pi1, pi2 = solution
            else:
                V[s], pi1, pi2 = fictitious_play(Q[s], FP_ITERS)

            delta = max(delta, np.max(np.abs(Q[s] - Q_old)))

        if it % 10 == 0:
            print(f"Q-iter {it}, delta = {delta:.6f}")

        # Convergence check
        if delta < TOL:
            print("Converged.")
            break

    return Q, V

# Run the car game
if __name__ == "__main__":
    env = CarGame(grid_size=3)
    Q, V = markov_game_q_iteration(env)

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
            value, pi1, pi2 = solution
        else:
            value, pi1, pi2 = fictitious_play(Q[s], FP_ITERS)


        is_ne = check_ne(Q[s], pi1, pi2, value)
        eps = max_deviation(Q[s], pi1, pi2, value)

        if is_ne:
            ne_true += 1
        #print(f"Epsilon = {eps:.6f} | State {s} is NE: {is_ne}")
        epsilons.append(eps)

    print("\nNE diagnostics:")
    print(f"NE states: {ne_true} / {len(env.states)}")
    print(f"Max epsilon: {max(epsilons):.6f}")
    print(f"Mean epsilon: {np.mean(epsilons):.6f}")
    

    