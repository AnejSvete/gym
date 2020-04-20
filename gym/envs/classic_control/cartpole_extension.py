import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
from scipy.constants import g, pi
from scipy.integrate import odeint

from shapely.geometry import LineString

# TODO
PUSH_CART_RIGHT = 0
PUSH_CART_LEFT = 1
PUSH_POLE_RIGHT = 2
PUSH_POLE_LEFT = 3
STOP_CART = 4
STOP_POLE = 5


class CartPoleMountainEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self, mode='train', de_solver='euler', seed=468731):

        self.seed(seed)

        self.world_width, self.world_height = 2 * pi, pi

        self.mode = mode

        self.gravity = -g
        self.mass_cart = 1.0
        self.mass_pole = 0.05
        self.total_mass = (self.mass_pole + self.mass_cart)
        self.length = 0.5  # actually half the pole's length
        self.pole_mass_length = (self.mass_pole * self.length)
        self.force_mag = 10
        self.tau = 0.01  # seconds between state updates

        self.de_solver = de_solver

        self.theta_min, self.theta_max = -pi / 2, pi / 2
        self.x_min, self.x_max = -self.world_width / 2, self.world_width / 2

        low = np.array([
            self.x_min * 2,
            -np.finfo(np.float32).max,
            self.theta_min * 2,
            -np.finfo(np.float32).max])
        high = np.array([
            self.x_max * 2,
            np.finfo(np.float32).max,
            self.theta_max * 2,
            np.finfo(np.float32).max])

        # self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.screen_width_pixels, self.screen_height_pixels = 1200, 600
        self.scale = self.screen_width_pixels / self.world_width

        self.cart_y_pixels = 100.0
        self.cart_width_pixels = 100.0
        self.cart_height_pixels = 60.0
        self.wheels_radius = self.cart_height_pixels / 3

        self.cart_y = self.cart_y_pixels / self.scale
        self.cart_width = self.cart_width_pixels / self.scale
        self.cart_height = self.cart_height_pixels / self.scale

        self.pole_width_pixels = 8.0
        self.pole_length = 2 * self.length
        self.pole_length_pixels = self.scale * self.pole_length

        # Environment parameters:
        self.initial_height = 1 / 3 * pi
        self.height = 0.5
        self.steepness = 1.7

        self.slope_length = pi / self.steepness

        self.bottom = self.x_min + self.slope_length
        self.bottom_width = pi / 32

        self.offset = pi / (2 * self.steepness) + self.bottom

        self.starting_position = self.bottom
        self.goal_position = self.x_max - self.world_width / 4

        self.viewer = None
        self.state = None

        self.episode_step = 0
        self.max_episode_steps = 2000

        self.min_goal, self.max_goal = \
            self.bottom + self.bottom_width / 2 + self.slope_length, self.x_max
        self.goal_stable_duration = 150
        self.times_at_goal = 0

    def reset(self, epoch=-1, num_epochs=-1):
        # print(f'ENV: epoch = {epoch}, num_epochs = {num_epochs}')
        if epoch == -1:
            if self.mode == 'train':
                self.state = np.random.uniform(
                    low=(self.x_min + pi / 16, -0.05, -pi / 6, -0.05),
                    high=(self.x_max - pi / 4, 0.05, pi / 6, 0.05),
                    size=(4,))
                # area = np.random.rand()
                # if area < 1 / 3:
                #     self.state = np.random.uniform(
                #         low=(self.bottom - 2 * self.bottom_width, -0.05, -pi / 6, -0.05),
                #         high=(self.bottom - 2 * self.bottom_width, 0.05, pi / 6, 0.05),
                #         size=(4,))
                # else:
                #     self.state = np.random.uniform(
                #         low=(self.min_goal, -0.05, -pi / 6, -0.05),
                #         high=(self.max_goal, 0.05, pi / 6, 0.05),
                #         size=(4,))
            else:
                self.state = np.random.uniform(
                    low=(self.bottom - self.bottom_width / 2,
                         -0.05, -0.3, -0.05),
                    high=(self.bottom + self.bottom_width / 2,
                          0.05, 0.3, 0.05),
                    size=(4,))
                # self.state = np.array([self.bottom, 0.0, 0.0, 0.0])
        else:
            if self.mode == 'train':
                # print(f'[{self.min_goal - (epoch / num_epochs) * 4 * pi}, {self.x_max - 2 * pi}]')
                # self.state = np.random.uniform()(low=(self.min_goal - (epoch / num_epochs) * 4 * pi, -0.05, -0.05, -0.05),
                #                                     high=(self.max_goal - 2 * pi, 0.05, 0.05, 0.05),
                #                                     size=(4,))
                # self.state = np.random.uniform()(low=(self.min_goal + pi, -0.05, -0.05, -0.05),
                #                                     high=(self.max_goal - 2 * pi, 0.05, 0.05, 0.05),
                #                                     size=(4,))
                # area = np.random.randint(0, 1 + 1)
                # if area == 0:
                area = np.random.rand()
                if area < 0.5:
                    self.state = np.random.uniform(low=(self.bottom - 1.5 * self.bottom_width, -0.05, -0.05, -0.05),
                                                    high=(self.bottom - 1.5 * self.bottom_width, 0.05, 0.05, 0.05),
                                                    size=(4,))
                else:
                    self.state = np.random.uniform(low=(self.min_goal + pi, -0.05, -0.05, -0.05),
                                                        high=(self.max_goal - pi, 0.05, 0.05, 0.05),
                                                        size=(4,))

            else:
                self.state = np.random.uniform(low=(self.bottom, -0.05, -0.05, -0.05),
                                                    high=(self.bottom, 0.05, 0.05, 0.05),
                                                    size=(4,))
        self.times_at_goal = 0
        self.episode_step = 0
        return np.array(self.state)

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def manual_control(self, cmd):
        s, s_dot, theta, theta_dot = self.state
        if cmd == STOP_CART:
            self.state = np.array([s, 0, theta, theta_dot])
        elif cmd == PUSH_CART_RIGHT:
            self.state = self.new_state(action=-1)
        elif cmd == PUSH_CART_LEFT:
            self.state = self.new_state(action=-2)
        elif cmd == STOP_POLE:
            self.state = np.array([s, s_dot, theta, 0])
        elif cmd == PUSH_POLE_RIGHT:
            self.state = np.array([s, s_dot, theta, theta_dot + 0.25])
        elif cmd == PUSH_POLE_LEFT:
            self.state = np.array([s, s_dot, theta, theta_dot - 0.25])

    def x(self, s):
        raise NotImplementedError

    def x_dot(self, s):
        raise NotImplementedError

    def x_dot_dot(self, s):
        raise NotImplementedError

    def y(self, s):
        raise NotImplementedError

    def y_dot(self, s):
        raise NotImplementedError

    def y_dot_dot(self, s):
        raise NotImplementedError

    def s_dot_dot(self, F):
        s, s_dot, theta, theta_dot = self.state
        return (-2 * F +
                self.y_dot(s) * (
                        -(self.gravity * (self.mass_pole + 2 * self.mass_cart + self.mass_pole * np.cos(2 * theta)))
                        - 2 * self.pole_length * self.mass_pole * np.cos(theta) * theta_dot ** 2
                        + s_dot ** 2 * (self.mass_pole * np.sin(2 * theta) * self.x_dot_dot(s)
                                        + (self.mass_pole + 2 * self.mass_cart
                                           + self.mass_pole * np.cos(2 * theta)) * self.y_dot_dot(s)))
                + self.x_dot(s) * (-(self.gravity * self.mass_pole * np.sin(2 * theta))
                                   - 2 * self.pole_length * self.mass_pole * np.sin(theta) * theta_dot ** 2
                                   + s_dot ** 2 * ((self.mass_pole + 2 * self.mass_cart
                                                    - self.mass_pole * np.cos(2 * theta)) * self.x_dot_dot(s)
                                                   + self.mass_pole * np.sin(2 * theta) * self.y_dot_dot(s)))) / \
               ((-self.mass_pole - 2 * self.mass_cart + self.mass_pole * np.cos(2 * theta)) * self.x_dot(s) ** 2
                - 2 * self.mass_pole * np.sin(2 * theta) * self.x_dot(s) * self.y_dot(s)
                - (self.mass_pole + 2 * self.mass_cart + self.mass_pole * np.cos(2 * theta)) * self.y_dot(s) ** 2)

    def theta_dot_dot(self, F):
        s, s_dot, theta, theta_dot = self.state
        return (F * (np.cos(theta) * self.x_dot(s) - np.sin(theta) * self.y_dot(s)) +
                (np.sin(theta) * self.x_dot(s) + np.cos(theta) * self.y_dot(s)) *
                (self.y_dot(s) * (-(self.pole_length * self.mass_pole * np.sin(theta) * theta_dot ** 2)
                                  + (self.mass_pole + self.mass_cart) * s_dot ** 2 * self.x_dot_dot(s))
                 + self.x_dot(s) * (self.pole_length * self.mass_pole * np.cos(theta) * theta_dot ** 2
                                    + (self.mass_pole + self.mass_cart) * (
                                            self.gravity - s_dot ** 2 * self.y_dot_dot(s))))) / \
               (self.pole_length * (
                       (-self.mass_pole - self.mass_cart + self.mass_pole * np.cos(theta) ** 2) * self.x_dot(s) ** 2
                       - self.mass_pole * np.sin(2 * theta) * self.x_dot(s) * self.y_dot(s)
                       - ((self.mass_pole + 2 * self.mass_cart
                           + self.mass_pole * np.cos(2 * theta)) * self.y_dot(s) ** 2) / 2.))

    def new_state(self, action):

        s, s_dot, theta, theta_dot = self.state

        if action in [-1, -2]:
            force = 10 * self.force_mag \
                if action == -1 else -10 * self.force_mag
        else:
            force = self.force_mag if action == 1 else -self.force_mag
            # force = self.force_mag * (int(action) - 1)
            # force = 0

        if self.de_solver == 'euler':

            s_dot_dot = self.s_dot_dot(force)[0]
            theta_dot_dot = self.theta_dot_dot(force)[0]

            s += self.tau * s_dot
            s_dot += self.tau * s_dot_dot
            theta += self.tau * theta_dot
            theta_dot += self.tau * theta_dot_dot

        elif self.de_solver == 'scipy':
            def ds(z, t, force=0.0):
                self.s_dot, self.s = z
                return np.array((self.s_dot_dot(force)[0], z[0]))

            def dtheta(z, t, force=0.0):
                self.theta_dot, self.theta = z
                return np.array((self.theta_dot_dot(force)[0], z[0]))

            t = np.linspace(0, self.tau, num=2)

            s_dot_tmp, s_tmp = odeint(ds, np.array([s_dot, s]),
                                      t, args=(force,)).T
            theta_dot_tmp, theta_tmp = odeint(dtheta,
                                              np.array([theta_dot, theta]),
                                              t, args=(force,)).T

            s_dot = s_dot_tmp[-1]
            s = s_tmp[-1]

            theta_dot = theta_dot_tmp[-1]
            theta = theta_tmp[-1]

        if theta < -pi:
            theta += 2 * pi
        elif theta > pi:
            theta -= 2 * pi

        return np.array([s, s_dot, theta, theta_dot])

    def reward(self, failed):
        s, s_dot, theta, theta_dot = self.state
        x = self.x(s)
        if failed:
            return -0.5
        else:
            return 0 if self.min_goal <= x <= self.max_goal \
                else -1 / (2 * self.max_episode_steps)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        s, s_dot, theta, theta_dot = self.new_state(action)
        self.state = (s, s_dot, theta, theta_dot)

        x, x_dot = self.x(s), self.x_dot(s)

        failed = not self.x_min <= x <= self.x_max or \
            not self.theta_min <= theta <= self.theta_max or \
            self.episode_step >= self.max_episode_steps - 1

        reward = self.reward(failed)

        if self.min_goal <= s <= self.max_goal:
            self.times_at_goal += 1
        else:
            self.times_at_goal = 0

        successful = self.times_at_goal >= self.goal_stable_duration

        done = failed or successful

        self.episode_step += 1

        info = {'success': successful,
                'time_limit': self.episode_step >= self.max_episode_steps}

        return np.array(self.state), reward, done, info

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
