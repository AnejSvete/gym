import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

from gym.envs.classic_control import rendering

import numpy as np
from scipy.constants import g, pi
from scipy.integrate import solve_ivp

from shapely.geometry import LineString, Polygon

# TODO
PUSH_CART_RIGHT = 0
PUSH_CART_LEFT = 1
PUSH_POLE_RIGHT = 2
PUSH_POLE_LEFT = 3
STOP_CART = 4
STOP_POLE = 5


class CartPoleExtensionEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self, mode='train', de_solver='scipy', seed=526245):

        self.gravity = -g
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_cart
        self.pole_length = 1.0
        self.force_mag = 10.0
        self.tau = 0.01  # seconds between state updates

        self.de_solver = de_solver
        self.mode = mode

        self.state = None
        self.viewer = None

        self.episode_step = 0
        self.max_episode_steps = 1000
        self.times_at_goal = 0
        self.goal_stable_duration = 10

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
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

    def s_dot_dot(self, F, s, s_dot, theta, theta_dot):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_2theta = np.cos(2 * theta)
        sin_2theta = np.sin(2 * theta)
        y_dot = self.y_dot(s)
        y_dot_dot = self.y_dot_dot(s)
        x_dot = self.x_dot(s)
        x_dot_dot = self.x_dot_dot(s)
        return (
            -2 * F + y_dot * (-(self.gravity * (
                self.mass_pole + 2 * self.mass_cart +
                self.mass_pole * cos_2theta))
                - 2 * self.pole_length * self.mass_pole *
                cos_theta * theta_dot ** 2
                + s_dot ** 2 * (self.mass_pole * sin_2theta * x_dot_dot
                                + (self.mass_pole + 2 * self.mass_cart
                                   + self.mass_pole * cos_2theta) * y_dot_dot))
            + x_dot * (-(self.gravity * self.mass_pole * sin_2theta)
                       - 2 * self.pole_length * self.mass_pole *
                       sin_theta * theta_dot ** 2
                       + s_dot ** 2 * (
                           (self.mass_pole + 2 * self.mass_cart
                            - self.mass_pole * cos_2theta) * x_dot_dot
                + self.mass_pole * sin_2theta * y_dot_dot))) / \
               ((-self.mass_pole - 2 * self.mass_cart +
                 self.mass_pole * cos_2theta) * x_dot ** 2
                - 2 * self.mass_pole * sin_2theta * x_dot * y_dot
                - (self.mass_pole + 2 * self.mass_cart +
                   self.mass_pole * cos_2theta) * y_dot ** 2
                )

    def theta_dot_dot(self, F, s, s_dot, theta, theta_dot):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_2theta = np.cos(2 * theta)
        sin_2theta = np.sin(2 * theta)
        y_dot = self.y_dot(s)
        y_dot_dot = self.y_dot_dot(s)
        x_dot = self.x_dot(s)
        x_dot_dot = self.x_dot_dot(s)
        return (F * (cos_theta * x_dot - sin_theta * y_dot) +
                (sin_theta * x_dot + cos_theta * y_dot) *
                (y_dot * (-(self.pole_length * self.mass_pole *
                            sin_theta * theta_dot ** 2)
                          + (self.mass_pole + self.mass_cart) *
                          s_dot ** 2 * x_dot_dot)
                 + x_dot * (self.pole_length * self.mass_pole *
                            cos_theta * theta_dot ** 2
                            + (self.mass_pole + self.mass_cart) * (
                                self.gravity - s_dot ** 2 * y_dot_dot)))) / \
               (self.pole_length * (
                   (-self.mass_pole - self.mass_cart + self.mass_pole *
                    cos_theta ** 2) * x_dot ** 2
                   - self.mass_pole * sin_2theta * x_dot * y_dot
                   - ((self.mass_pole + 2 * self.mass_cart
                       + self.mass_pole * cos_2theta) * y_dot ** 2) / 2.))

    def new_state(self, action):

        s, s_dot, theta, theta_dot = self.state

        if action in [-1, -2]:
            force = 10 * self.force_mag \
                if action == -1 else -10 * self.force_mag
        else:
            force = self.action_to_force(action)

        if self.de_solver == 'euler':

            s_dot_dot = self.s_dot_dot(force, s, s_dot, theta, theta_dot)
            theta_dot_dot = self.theta_dot_dot(
                force, s, s_dot, theta, theta_dot)

            s += self.tau * s_dot
            s_dot += self.tau * s_dot_dot
            theta += self.tau * theta_dot
            theta_dot += self.tau * theta_dot_dot

        elif self.de_solver == 'scipy':
            def Fs(t, y, force):
                return np.array((y[1], self.s_dot_dot(
                    force, y[0], y[1], theta, theta_dot)))

            def Ftheta(t, y, force):
                return np.array((y[1], self.theta_dot_dot(
                    force, s, s_dot, y[0], y[1])))

            ss, s_dots = solve_ivp(
                Fs, (0, self.tau), np.array([s, s_dot]),
                args=(force,), method='DOP853').y
            thetas, theta_dots = solve_ivp(
                Ftheta, (0, self.tau), np.array([theta, theta_dot]),
                args=(force,), method='DOP853').y

            s_dot = s_dots[-1]
            s = ss[-1]

            theta_dot = theta_dots[-1]
            theta = thetas[-1]

        if theta < -pi:
            theta += 2 * pi
        elif theta > pi:
            theta -= 2 * pi

        return np.array([s, s_dot, theta, theta_dot])

    def action_to_force(self, action):
        raise NotImplementedError

    def reward(self, failed):
        raise NotImplementedError

    def in_goal_state(self):
        raise NotImplementedError

    def has_failed(self, x, theta):
        raise NotImplementedError

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        s, s_dot, theta, theta_dot = self.new_state(action)
        self.state = np.array([s, s_dot, theta, theta_dot])

        x, x_dot = self.x(s), self.x_dot(s)

        failed = self.has_failed(x, theta)

        reward = self.reward(failed)

        if self.in_goal_state():
            self.times_at_goal += 1
        else:
            self.times_at_goal = 0

        successful = self.times_at_goal >= self.goal_stable_duration

        done = failed or successful

        self.episode_step += 1

        info = {'success': successful,
                'time_limit': self.episode_step >= self.max_episode_steps}

        return self.obeservation(), reward, done, info

    def obeservation(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
