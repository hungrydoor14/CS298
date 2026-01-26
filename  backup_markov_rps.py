import numpy as np
from collections import defaultdict

GAMMA = 0.9
FP_ITERS = 8000
TOL = 1e-6

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

def fictitious_play(Q, iters=FP_ITERS):
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

def solve_rps():
    """
    Stateless Markov game
    S has 1 state
    """
    # R, P, S
    Q = np.array([
        [ 0, -1,  1],
        [ 1,  0, -1],
        [-1,  1,  0]
    ])

    # run fictitious play
    value, pi1, pi2 = fictitious_play(Q)

    print("ROCK PAPER SCISSORS")
    print("Value:", value)
    print("Player 1 policy:", pi1)
    print("Player 2 policy:", pi2)
    print("Expected: uniform [1/3, 1/3, 1/3]\n")

if __name__ == "__main__":
    solve_rps()