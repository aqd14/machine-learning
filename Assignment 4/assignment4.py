import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

class Environment(object):
    # Implement easy21
    # state: (dealer's first card, player's hand)
    # action: 1(hit); 0(stick)
    # reward: -1(lose); 0(draw); 1(win)

    def __init__(self):
        self.new_game()

    def new_game(self):
        self.dealer_first_card = self.dealer_hand = np.random.randint(1, 11)
        self.player_hand = np.random.randint(1, 11)
        self.game_end = False

    def deal_card(self):
        color = np.random.randint(3)
        value = np.random.randint(1, 11)
        if color <= 1:
            return value
        else:
            return -value

    def observe(self):
        # return current state
        return [self.dealer_first_card, self.player_hand]

    def is_terminal(self):
        return self.game_end

    def step(self, action):
        if self.is_terminal():
            self.new_game()
        if action == 1:
            # Player hits
            self.player_hand += self.deal_card()
            if 1 < self.player_hand < 22:
                # continue for another action, zero reward
                reward = 0
                self.game_end = False
            else:
                # Player goes bust
                self.player_hand = 0
                reward = -1
                self.game_end = True
        else:
            # Player sticks
            while self.dealer_hand < 17:
                # Dealer hits
                self.dealer_hand += self.deal_card()
                if 1 < self.dealer_hand < 22:
                    continue
                else:
                    # Dealer goes bust
                    self.dealer_hand = 0
                    reward = 1
                    self.game_end = True
                    return [self.dealer_first_card, self.player_hand], reward
            # Dealer sticks
            if self.dealer_hand > self.player_hand:
                reward = -1
            elif self.dealer_hand == self.player_hand:
                reward = 0
            else:
                reward = 1
            self.game_end = True
        return [self.dealer_first_card, self.player_hand], reward

def epsilon_greedy(N0, N, Q, x, y):
    # epsilon-greedy exploration
    e = N0 / (N0 + np.sum(N[x - 1, y - 1, :]))
    if np.random.uniform(0, 1) > e:
        action = np.argmax(Q[x - 1, y - 1, :])
    else:
        action = np.random.randint(0, 2)
    return action

def monte_carlo(max_episode, discount, N0):
    # Initialization
    Q = np.zeros([10, 21, 2])
    N = np.zeros([10, 21, 2])
    for i in range(max_episode):
        # initial a new episode
        episode = Environment()
        # the initial state of the episode
        x, y = episode.observe()
        # sample until terminal
        history = []
        while not episode.is_terminal():
            # decide action
            action = epsilon_greedy(N0, N, Q, x, y)
            N[x - 1, y - 1, action] += 1
            # run one step
            (state, reward) = episode.step(action)
            history.append(([x, y], action, reward))
            [x, y] = state
        # calculate return Gt for each state in this episode
        Gt = 0
        for j, (state, action, reward) in enumerate(reversed(history)):
            [x, y] = state
            alpha = 1.0 / N[x - 1, y - 1, action]
            Gt = discount * Gt + reward
            Q[x - 1, y - 1, action] += alpha * (Gt - Q[x - 1, y - 1, action])
    return Q

def plot_value_function(V):
    # plot value function
    x = np.arange(1, 11)
    y = np.arange(1, 22)
    xs, ys = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_ylabel('Player sum')
    ax.set_xlabel('Dealer showing')
    ax.plot_wireframe(xs, ys, V.T, rstride=1, cstride=1)
    plt.show()

def get_value_function(Q):
    return np.amax(Q, axis=2)

def main():
    N0 = 100
    discount = 1 # gamma
    max_episode = 500000
    Q = monte_carlo(max_episode, discount, N0)
    # optimal value function
    V_max = get_value_function(Q)
    plot_value_function(V_max)

if __name__ == "__main__":
    main()