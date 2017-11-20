import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def step(s,a):
    """
    Implements one step for the Easy21 game. a (action) is 0 for hit, 1 for stick. s (state) is of the form (dealer's card,player's sum).
    returns (s',r) where s' is the next state and r is the reward. if s'==-1 then that is the terminal state.
    red cards are -1, black cards are +1
    """
    if (a==0):
        draw = [np.random.choice([-1,1,1]),np.random.randint(1,11)] #draw appropriate card with appropriate colour
        #print draw
        s[1] += draw[0]*draw[1] #player did not go bust, return s and zero reward
        r = 0
        if (s[1]>21 or s[1]<1): #player is bust, terminal-state and -1 reward
            #print "player went bust"
            s[0] = -1
            r = -1
    else:
        while (s[0]<17 and s[0]>=1):
            draw = [np.random.choice([-1,1,1]),np.random.randint(1,11)] #draw appropriate card with appropriate colour
            #print draw
            s[0] += draw[0]*draw[1]
        if (s[0]>21 or s[0]<1):  #dealer is bust, terminal-state
            #print "dealer went bust"
            r = +1
        else: #dealer did not go bust, now the sum will be compared
            #print "dealer did not go bust, comparing cards"
            if (s[0]>s[1]): r = -1
            elif (s[0]<s[1]): r = 1
            else: r = 0

        s[0] = -1
    #print "state,reward is",s,r
    return s,r

"""
So my actions are hit (0) or stick (1)
My state is actually [dealer's card,player's sum]
Rewards are +1,0,-1
"""
num_states = 210.0  # 21*10
num_actions = 2.0
Q = np.zeros((10, 21, 2))  # action value function
V = np.zeros((10, 21))  # state value function
# note that policy will not be explicitly maintained in Monte-Carlo control

Nsa = np.zeros_like(Q)  # number of times chosen state-action pair
Ns = np.zeros_like(V)  # number of times visited state
N0 = 100.0


def policy(s, Q, Nst):  # return action according to epsilon-greedy policy on Q
    greedy_action = np.argmax(Q[s[0] - 1, s[1] - 1])
    epsilon = N0 / (N0 + Nst)
    a = np.random.choice([greedy_action, 0, 1], p=[1 - epsilon, epsilon / 2.0, epsilon / 2.0])
    return a


def run_episode():
    history = []
    state = [np.random.randint(1, 11), np.random.randint(1, 11)]
    old_state = []
    # print "Initial state is",state
    while (state[0] != -1):  # capture an entire episode of experience
        a = policy(state, Q, Ns[state[0] - 1, state[1] - 1])  # always 0 initially
        # print "The action chosen is (0-hit,1-stick)",a
        old_state[:] = state
        temp, r = step(state, a)
        history.append([old_state[:], a, r])

    # print "History accumulated is",history

    reward = 0  # now calculating g - returns
    g = []
    for i in range(len(history) - 1, -1, -1):
        reward += history[i][2]
        # print "Reward",i,reward
        g.append(reward)

    i = 1
    for s, a, r in history:
        Ns[s[0] - 1, s[1] - 1] += 1
        Nsa[s[0] - 1, s[1] - 1, a] += 1
        # print Ns[s[0]-1,s[1]-1],Nsa[s[0]-1,s[1]-1,a]
        Q[s[0] - 1, s[1] - 1, a] = Q[s[0] - 1, s[1] - 1, a] + (1.0 / Nsa[s[0] - 1, s[1] - 1, a]) * (
            g[-i] - Q[s[0] - 1, s[1] - 1, a])
        i += 1

    return Q, history


def get_value_function(Q):
    V = np.zeros((Q.shape[0], Q.shape[1]))
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            V[i][j] = np.max(Q[i][j])
    return V

def create_surf_plot(X, Y, Z, fig_idx=1):
  fig = plt.figure(fig_idx)
  ax = fig.add_subplot(111, projection="3d")

  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
  # surf = ax.plot_wireframe(X, Y, Z)

  return surf

def plot_value_function(V):

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(range(V.shape[1]), range(V.shape[0]))
    # ax.plot_wireframe(X, Y, V)
    # ax.set_xlabel('Sum of player cards')
    # ax.set_ylabel('Initial dealer card')
    # plt.show()
    DEALER_RANGE = range(1, 11)
    PLAYER_RANGE = range(1, 22)

    V = np.max(Q, axis=2)
    X, Y = np.mgrid[DEALER_RANGE, PLAYER_RANGE]

    surf = create_surf_plot(X, Y, V)

    plt.title("V*")
    plt.ylabel('player sum', size=18)
    plt.xlabel('dealer', size=18)
    plt.show()
    plt.clf()

if __name__ == "__main__":
    Q, history = run_episode()
    V = get_value_function(Q)
    plot_value_function(V)
    '''
    state = [np.random.randint(1,11),np.random.randint(1,11)]
    action = 0
    state,reward = step(state,action)
    '''


