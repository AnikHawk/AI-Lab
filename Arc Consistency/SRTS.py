#!/usr/bin/env python
# coding: utf-8

# # Sutton and Barto Racetrack: Sarsa
# Exercise 5.8 from *Reinforcement Learning: An Introduction* by Sutton and Barto.
# 
# This notebook applies the **Sarsa** algorithm from Chapter 6 to the Racetrack problem from Chapter 5. 
# 
# Python Notebook by Patrick Coady: [Learning Artificial Intelligence](https://learningai.io/)

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


class RaceTrack(object):
    """
    RaceTrack object maintains and updates the race track 
    state. Interaction with the class is through
    the take_action() method. The take_action() method returns
    a successor state and reward (i.e. s' and r)

    The class constructor is given a race course as a list of 
    strings. The constructor loads the course and initializes 
    the environment state.
    """

    def __init__(self, course):
        """
        Load race course, set any min or max limits in the 
        environment (e.g. max speed), and set initial state.
        Initial state is random position on start line with 
        velocity = (0, 0).

        Args:
            course: List of text strings used to construct
                race-track.
                    '+': start line
                    '-': finish line
                    'o': track
                    'X': wall

        Returns:
            self
        """
        self.NOISE = 0.0
        self.EPS = 0.1  # epsilon-greedy coefficient
        self.MAX_VELOCITY = 4
        self.start_positions = []
        self.course = None
        self._load_course(course)
        self._random_start_position()
        self.velocity = np.array([0, 0], dtype=np.int16)

    def take_action(self, action):
        """
        Take action, return state' and reward

        Args:
            action: 2-tuple of requested change in velocity in x- and
                y-direction. valid action is -1, 0, +1 in each axis.

        Returns:
            reward: integer
        """

        self._update_velocity(action)
        self._update_position()
        if self.is_terminal_state():
            return 100.0

        return -1.0

    def get_state(self):
        """Return 2-tuple: (position, velocity). Each is a 2D numpy array."""
        return self.position.copy(), self.velocity.copy()

    def _update_velocity(self, action):
        """
        Update x- and y-velocity. Clip at 0 and self.MAX_VELOCITY

        Args:
            action: 2-tuple of requested change in velocity in x- and
                y-direction. valid action is -1, 0, +1 in each axis.        
        """
        if np.random.rand() > self.NOISE:
            self.velocity += np.array(action, dtype=np.int16)
            self.velocity = np.minimum(self.velocity, self.MAX_VELOCITY)
            self.velocity = np.maximum(self.velocity, 0)

    def reset(self):
        self._random_start_position()
        self.velocity = np.array([0, 0], dtype=np.int16)

    def _update_position(self):
        """
        Update position based on present velocity. Check at fine time 
        scale for wall or finish. If wall is hit, set position to random
        position at start line. If finish is reached, set position to 
        first crossed point on finish line.
        """
        for tstep in range(0, self.MAX_VELOCITY + 1):
            t = tstep / self.MAX_VELOCITY
            pos = self.position + np.round(self.velocity * t).astype(np.int16)
            if self._is_wall(pos):
                self._random_start_position()
                self.velocity = np.array([0, 0], dtype=np.int16)
                return
            if self._is_finish(pos):
                self.position = pos
                self.velocity = np.array([0, 0], dtype=np.int16)
                return
        self.position = pos

    def _random_start_position(self):
        """Set car to random position on start line"""
        self.position = np.array(random.choice(self.start_positions),
                                 dtype=np.int16)

    def _load_course(self, course):
        """Load course. Internally represented as numpy array"""
        y_size, x_size = len(course), len(course[0])
        self.course = np.zeros((x_size, y_size), dtype=np.int16)
        for y in range(y_size):
            for x in range(x_size):
                point = course[y][x]
                if point == 'o':
                    self.course[x, y] = 1
                elif point == '-':
                    self.course[x, y] = 0
                elif point == '+':
                    self.course[x, y] = 2
                elif point == 'W':
                    self.course[x, y] = -1
        # flip left/right so (0,0) is in bottom-left corner
        self.course = np.fliplr(self.course)
        for y in range(y_size):
            for x in range(x_size):
                if self.course[x, y] == 0:
                    self.start_positions.append((x, y))

    def _is_wall(self, pos):
        """Return True is position is wall"""
        return self.course[pos[0], pos[1]] == -1

    def _is_finish(self, pos):
        """Return True if position is finish line"""
        return self.course[pos[0], pos[1]] == 2

    def is_terminal_state(self):
        """Return True at episode terminal state"""
        return (self.course[self.position[0],
                            self.position[1]] == 2)

    def action_to_tuple(self, a):
        """Convert integer action to 2-tuple: (ax, ay)"""
        ax = a // 3 - 1
        ay = a % 3 - 1

        return ax, ay

    def tuple_to_action(self, a):
        """Convert 2-tuple to integer action: {0-8}"""
        return int((a[0] + 1) * 3 + a[1] + 1)

    def greedy_eps(self, Q):
        """Based on state and Q values, return epsilon-greedy action"""
        s = self.get_state()
        s_x, s_y = s[0][0], s[0][1]
        s_vx, s_vy = s[1][0], s[1][1]
        if np.random.rand() > self.EPS:
            print(Q[s_x, s_y, s_vx, s_vy, :, :])
            if (np.max(Q[s_x, s_y, s_vx, s_vy, :, :]) ==
                    np.min(Q[s_x, s_y, s_vx, s_vy, :, :])):
                a = (0, 0)
            else:
                a = np.argmax(Q[s_x, s_y, s_vx, s_vy, :, :])
                a = np.unravel_index(a, (3, 3)) - np.array([1, 1])
                a = (a[0], a[1])
        else:
            a = self.action_to_tuple(random.randrange(9))

        return a

    def srts(self,Q):
        pass


    def state_action(self, s, a):
        """Build state-action tuple for indexing Q NumPy array"""
        s_x, s_y = s[0][0], s[0][1]
        s_vx, s_vy = s[1][0], s[1][1]
        a_x, a_y = a[0] + 1, a[1] + 1
        s_a = (s_x, s_y, s_vx, s_vy, a_x, a_y)

        return s_a

    # In[3]:


# Race Track from Sutton and Barto Figure 5.6

big_course = ['WWWWWWWWWWWWWWWWWW',
              'WWWWooooooooooooo+',
              'WWWoooooooooooooo+',
              'WWWoooooooooooooo+',
              'WWooooooooooooooo+',
              'Woooooooooooooooo+',
              'Woooooooooooooooo+',
              'WooooooooooWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWWooooooWWWWWWWW',
              'WWWWooooooWWWWWWWW',
              'WWWW------WWWWWWWW']

# Tiny course for debug

tiny_course = ['WWWWWW',
               'Woooo+',
               'Woooo+',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'W--WWW', ]

# In[4]:


# Problem Initialization

course = big_course
x_size, y_size = len(course[0]), len(course)
# Q[x_pos, y_pos, x_velocity, y-velocity, x-acceleration, y-acceleration]
Q = np.zeros((x_size, y_size, 5, 5, 3, 3), dtype=np.float64)
position_map = np.zeros((x_size, y_size), dtype=np.float64)  # track explored positions

N = 2000  # num episodes
gamma = 1.0
alpha = 0.1
track = RaceTrack(course)

# Sarsa

epochs = []
counts = []
count = 0
for e in range(N):
    if (e + 1) % 200 == 0: print('Episode {}'.format(e + 1))
    track.reset()
    s = track.get_state()
    a = track.greedy_eps(Q)

    while not track.is_terminal_state():
        position_map[s[0][0], s[0][1]] += 1
        count += 1
        r = track.take_action(a)
        s_prime = track.get_state()
        a_prime = track.greedy_eps(Q)
        s_a = track.state_action(s, a)
        s_a_prime = track.state_action(s_prime, a_prime)
        Q[s_a] = Q[s_a] + alpha * (r + gamma * Q[s_a_prime] - Q[s_a])
        s, a = s_prime, a_prime
    epochs.append(e)
    counts.append(count)





# In[5]:


plt.plot(epochs, counts)
plt.title('Simulation Steps vs. Episodes')
plt.xlabel('Epochs')
plt.ylabel('Total Simulation Steps')
plt.show()

# In[6]:


print('Heat map of position exploration:')
plt.imshow(np.flipud(position_map.T), cmap='hot', interpolation='nearest')
plt.show()

# In[7]:


# Convert Q (action-values) to pi (policy)
pi = np.zeros((x_size, y_size, 5, 5), dtype=np.int16)
for idx in np.ndindex(x_size, y_size, 5, 5):
    a = np.argmax(Q[idx[0], idx[1], idx[2], idx[3], :, :])
    a = np.unravel_index(a, (3, 3))
    pi[idx] = track.tuple_to_action(a - np.array([1, 1]))

# In[8]:


# Run learned policy on test case

pos_map = np.zeros((x_size, y_size))
track.reset()
for e in range(1000):
    s = track.get_state()
    s_x, s_y = s[0][0], s[0][1]
    s_vx, s_vy = s[1][0], s[1][1]
    pos_map[s_x, s_y] += 1  # exploration map
    act = track.action_to_tuple(pi[s_x, s_y, s_vx, s_vy])
    track.take_action(act)
    if track.is_terminal_state(): break

print('Sample trajectory on learned policy:')
pos_map = (pos_map > 0).astype(np.float32)
pos_map += track.course  # overlay track course
plt.imshow(np.flipud(pos_map.T), cmap='hot', interpolation='nearest')
plt.show()

