import numpy as np

TIMEOUT_AMOUNT = 150
NUM_ACTIONS = 32
STEP_SIZE = 0.05
MAX_MOVE = 0.05 # theroetically None or [0,1.414]
T = 50
SET_HOUSE_POSITIONS = True

class DogGame:
    def __init__(self, N_actions=8, step_size=0.05):
        self.N = N_actions
        self.step = step_size
        self.angle_step = 2 * np.pi / self.N

        # max_move = None -> unlimited (to boundary)
        # max_move = float -> limited step size (to 1^2 or 1.414)
        self.max_move = MAX_MOVE

        if SET_HOUSE_POSITIONS:
            self.blue_house = [0.25, 0.25]
            self.red_house  = [0.75, 0.75]
        else:
            self.blue_house = np.random.rand(2)
            self.red_house  = np.random.rand(2)

    def random_state(self):
        self.blue = np.random.rand(2)
        self.red  = np.random.rand(2)

        self.dog = (self.blue + self.red) / 2.0
        self.t = 0
        return self.get_state()

    def get_state(self):
        return self.dog.copy()

    def action_to_angle(self, a):
        return a * self.angle_step

    def step_game(self, a_blue, a_red):

        self.t += 1

        # Convert actions to direction vectors 
        theta_b = self.action_to_angle(a_blue)
        theta_r = self.action_to_angle(a_red)

        dir_blue = np.array([np.cos(theta_b), np.sin(theta_b)])
        dir_red  = np.array([np.cos(theta_r), np.sin(theta_r)])

        # Move player
        self.blue = self._move_player(self.blue, dir_blue)
        self.red  = self._move_player(self.red, dir_red)

        # Dog jumps to midpoint 
        self.dog = (self.blue + self.red) / 2.0

        done, r_b, r_r = self.check_terminal()

        return self.get_state(), r_b, r_r, done


    def check_terminal(self):
        if self.t >= TIMEOUT_AMOUNT:
            return True, 0.0, 0.0

        dist_blue = np.linalg.norm(self.dog - self.blue_house)
        dist_red  = np.linalg.norm(self.dog - self.red_house)

        if dist_blue < 0.05:
            return True, 1.0, -1.0
        if dist_red < 0.05:
            return True, -1.0, 1.0

        return False, 0.0, 0.0

    def solve_equilibrium(self):

        best_red_value = -np.inf
        best_red_action = None
        best_blue_action = None

        # --- RED chooses action anticipating blue sabotage ---
        for a_r in range(self.N):

            worst_case = np.inf

            for a_b in range(self.N):

                # Convert actions to direction vectors
                theta_b = self.action_to_angle(a_b)
                theta_r = self.action_to_angle(a_r)

                dir_blue = np.array([np.cos(theta_b), np.sin(theta_b)])
                dir_red  = np.array([np.cos(theta_r), np.sin(theta_r)])

                # Simulate REAL movement 
                blue_sim = self._move_player(self.blue, dir_blue)
                red_sim  = self._move_player(self.red, dir_red)

                dog_sim = (blue_sim + red_sim) / 2.0

                # Compute payoff
                dist_red  = np.linalg.norm(dog_sim - self.red_house)
                dist_blue = np.linalg.norm(dog_sim - self.blue_house)

                payoff = dist_blue - dist_red  # zero-sum

                worst_case = min(worst_case, payoff)

            if worst_case > best_red_value:
                best_red_value = worst_case
                best_red_action = a_r

        # --- BLUE best response to chosen red ---
        best_blue_value = np.inf

        for a_b in range(self.N):

            theta_b = self.action_to_angle(a_b)
            theta_r = self.action_to_angle(best_red_action)

            dir_blue = np.array([np.cos(theta_b), np.sin(theta_b)])
            dir_red  = np.array([np.cos(theta_r), np.sin(theta_r)])

            blue_sim = self._move_player(self.blue, dir_blue)
            red_sim  = self._move_player(self.red, dir_red)

            dog_sim = (blue_sim + red_sim) / 2.0

            dist_red  = np.linalg.norm(dog_sim - self.red_house)
            dist_blue = np.linalg.norm(dog_sim - self.blue_house)

            payoff = dist_blue - dist_red

            if payoff < best_blue_value:
                best_blue_value = payoff
                best_blue_action = a_b

        return best_blue_action, best_red_action
    

    def _move_player(self, pos, direction):
        direction = direction / np.linalg.norm(direction)

        if self.max_move is None:
            # UNLIMITED: move to boundary
            t_candidates = []

            for i in range(2):
                if direction[i] > 0:
                    t_candidates.append((1 - pos[i]) / direction[i])
                elif direction[i] < 0:
                    t_candidates.append((0 - pos[i]) / direction[i])

            if len(t_candidates) == 0:
                return pos

            t_max = min(t_candidates)
            new_pos = pos + t_max * direction

        else:
            # LIMITED: move fixed distance
            new_pos = pos + self.max_move * direction

        return np.clip(new_pos, 0, 1)

def build_payoff_matrix(env):
    M = np.zeros((env.N, env.N))

    dog = env.dog.copy()

    for a_b in range(env.N):
        for a_r in range(env.N):

            theta_b = env.action_to_angle(a_b)
            theta_r = env.action_to_angle(a_r)

            theta_d = (theta_b + theta_r) / 2.0
            move = np.array([np.cos(theta_d), np.sin(theta_d)]) * env.step

            new_pos = dog + move

            for i in range(2):
                if new_pos[i] < 0:
                    new_pos[i] = -new_pos[i]
                elif new_pos[i] > 1:
                    new_pos[i] = 2 - new_pos[i]

            dist_red  = np.linalg.norm(new_pos - env.red_house)
            dist_blue = np.linalg.norm(new_pos - env.blue_house)

            M[a_b, a_r] = dist_blue - dist_red

    return M

def rollout(env, T=T):
    traj = []

    env.random_state()

    for _ in range(T):

        a_b, a_r = env.solve_equilibrium()
        old_blue = env.blue.copy()
        old_red = env.red.copy()
        old_dog = env.dog.copy()

        _, _, _, done = env.step_game(a_b, a_r)

        traj.append((
            old_dog,
            env.dog.copy(),
            old_blue,
            env.blue.copy(),
            old_red,
            env.red.copy()
        ))


        if done:
            break

    return traj

def draw_trajectory(ax, traj, env):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    blue_house = env.blue_house
    red_house  = env.red_house

    ax.scatter(*blue_house, color="blue", s=200, marker="s", label="Blue House")
    ax.scatter(*red_house,  color="red",  s=200, marker="s", label="Red House")

    for (old_dog, new_dog, old_blue, new_blue, old_red, new_red) in traj:

        # Dog movement
        ax.arrow(old_dog[0], old_dog[1],
                new_dog[0] - old_dog[0],
                new_dog[1] - old_dog[1],
                color="black", head_width=0.02)

        # Blue movement
        ax.arrow(old_blue[0], old_blue[1],
                new_blue[0] - old_blue[0],
                new_blue[1] - old_blue[1],
                color="blue", head_width=0.02)

        # Red movement
        ax.arrow(old_red[0], old_red[1],
                new_red[0] - old_red[0],
                new_red[1] - old_red[1],
                color="red", head_width=0.02)



if __name__ == "__main__":

    env = DogGame(N_actions=NUM_ACTIONS, step_size=STEP_SIZE)

    # 1) Statistical Evaluation

    blue_wins = 0
    red_wins = 0
    timeouts = 0

    for _ in range(1000):
        state = env.random_state()
        done = False

        while not done:
            a_b = np.random.randint(env.N)
            a_r = np.random.randint(env.N)

            state, r_b, r_r, done = env.step_game(a_b, a_r)

        if r_b == 1:
            blue_wins += 1
        elif r_r == 1:
            red_wins += 1
        else:
            timeouts += 1

    print("Blue wins:", blue_wins)
    print("Red wins:", red_wins)
    print("Timeouts:", timeouts)

   # 2) Interactive Visualization
    from matplotlib.widgets import Button
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    def update_plot(event=None):
        new_env = DogGame(N_actions=NUM_ACTIONS, step_size=STEP_SIZE)
        traj = rollout(new_env)
        draw_trajectory(ax, traj, new_env)
        fig.canvas.draw_idle()

    axnext = plt.axes([0.4, 0.05, 0.2, 0.075])
    bnext = Button(axnext, "New Rollout")
    bnext.on_clicked(update_plot)

    update_plot()
    plt.show()
