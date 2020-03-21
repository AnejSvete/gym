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
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.world_width, self.world_height = 4 * pi, 2 * pi

        self.gravity = -g
        self.mass_cart = 10.0
        self.mass_pole = 5.0
        self.total_mass = (self.mass_pole + self.mass_cart)
        self.length = 1.0  # actually half the pole's length
        self.pole_mass_length = (self.mass_pole * self.length)
        self.force_mag = 100
        self.tau = 0.02  # seconds between state updates

        self.de_solver = 'scipy'

        self.theta_min, self.theta_max = -pi / 3, pi / 3
        self.x_min, self.x_max = -self.world_width / 2, self.world_width / 2

        low = np.array([self.x_min * 2, -np.finfo(np.float32).max, self.theta_min * 2, -np.finfo(np.float32).max])
        high = np.array([self.x_max * 2, np.finfo(np.float32).max, self.theta_max * 2, np.finfo(np.float32).max])

        # self.action_space = spaces.Discrete(2)
        # self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.screen_width_pixels, self.screen_height_pixels = 1600, 800
        self.scale = self.screen_width_pixels / self.world_width

        self.cart_y_pixels = 100
        self.cart_width_pixels = 100.0
        self.cart_height_pixels = 60.0
        self.wheels_radius = self.cart_height_pixels / 3

        self.cart_y = self.cart_y_pixels / self.scale
        self.cart_width = self.cart_width_pixels / self.scale
        self.cart_height = self.cart_height_pixels / self.scale

        self.pole_width_pixels = 20.0
        self.pole_length = 2 * self.length
        self.pole_length_pixels = self.scale * self.pole_length

        self.height = 1
        self.steepness = 0.8
        self.initial_height = 0.75 * pi

        # self.goal_position = 3 / 2 * pi
        self.goal_position = 1.25 * pi

        self.seed()
        self.viewer = None
        self.state = None

        self.episode_step = 1
        self.max_episode_steps = 500

        self.goal_margin = 1 / 24
        self.goal_stable_duration = 25
        self.times_at_goal = 0

    def reset(self):
        # self.state = self.np_random.uniform(low=(-9 * pi / 16, -0.05, -0.05, -0.05),
        #                                     high=(-7 * pi / 16, 0.05, 0.05, 0.05),
        #                                     size=(4,))
        self.state = np.array([- 9 * pi / 16, 0.0, 0.0, 0.0])
        # self.state = np.array([- pi / 2, 0.0, 0.0, 0.0])
        self.times_at_goal = 0
        self.episode_step = 1
        return np.array(self.state)

    def seed(self, seed=None):
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
        return s

    def x_dot(self, s):
        return 1

    def x_dot_dot(self, s):
        return 0

    def y(self, s):
        return self.height * np.sin(self.steepness * (np.clip(s, -15 / 8 * pi, 5 * pi / 8))) + self.initial_height
        # return self.height * np.sin(self.steepness * (np.clip(s, -7 / 4 * pi, 3 * pi / 4) - pi / 8)) + self.initial_height
        # return self.height * np.sin(self.steepness * (np.clip(s, -7 / 4 * pi, 11 * pi / 12) - pi / 4)) + self.initial_height
        # return self.height * np.sin(self.steepness * (np.clip(s, -3 / 2 * pi, 7 * pi / 6) - pi / 2)) + self.initial_height

    def y_dot(self, s):
        return self.height * self.steepness * np.cos(self.steepness * (np.clip(s, -15 / 8 * pi, 5 * pi / 8)))
        # return self.height * self.steepness * np.cos(self.steepness * (np.clip(s, -7 / 4 * pi, 3 * pi / 4) - pi / 8))
        # return self.height * self.steepness * np.cos(self.steepness * (np.clip(s, -7 / 4 * pi, 11 * pi / 12) - pi / 4))
        # return self.height * self.steepness * np.cos(self.steepness * (np.clip(s, -3 / 2 * pi, 7 * pi / 6) - pi / 2))

    def y_dot_dot(self, s):
        return -self.height * self.steepness ** 2 * np.sin(self.steepness * (np.clip(s, -15 / 8 * pi, 5 * pi / 8)))
        # return -self.height * self.steepness**2 * np.sin(self.steepness * (np.clip(s, -7 / 4 * pi, 3 * pi / 4) - pi / 8))
        # return -self.height * self.steepness**2 * np.sin(self.steepness * (np.clip(s, -7 / 4 * pi, 11 * pi / 12) - pi / 4))
        # return -self.height * self.steepness**2 * np.sin(self.steepness * (np.clip(s, -3 / 2 * pi, 7 * pi / 6) - pi / 2))

    def phi(self, t):
        return np.arctan(self.y_dot(t))

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
            force = 10 * self.force_mag if action == -1 else -10 * self.force_mag
        else:
            force = self.force_mag if action == 1 else -self.force_mag
            # force = self.force_mag * (int(action) - 1)

        if self.de_solver == 'euler':

            s_dot_dot = self.s_dot_dot(force)
            theta_dot_dot = self.theta_dot_dot(force)

            s += self.tau * s_dot
            s_dot += self.tau * s_dot_dot
            theta += self.tau * theta_dot
            theta_dot += self.tau * theta_dot_dot

        elif self.de_solver == 'scipy':
            def ds(z, t, force=0.0):
                self.s_dot, self.s = z
                return np.array((self.s_dot_dot(force), z[0]))

            def dtheta(z, t, force=0.0):
                self.theta_dot, self.theta = z
                return np.array((self.theta_dot_dot(force), z[0]))

            t = np.linspace(0, 0.02, num=2)

            s_dot_tmp, s_tmp = odeint(ds, np.array([s_dot, s]), t, args=(force,)).T
            theta_dot_tmp, theta_tmp = odeint(dtheta, np.array([theta_dot, theta]), t, args=(force,)).T

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
            return - 2 * (self.max_episode_steps - self.episode_step)
            # return -1
        else:
            return 0 if np.abs(x - self.goal_position) < self.goal_margin * self.world_width else -1
            # return 1 if np.abs(x - self.goal_position) < self.goal_margin * self.world_width else 0.0

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        s, s_dot, theta, theta_dot = self.new_state(action)
        self.state = (s, s_dot, theta, theta_dot)

        x, x_dot = self.x(s), self.x_dot(s)

        failed = not self.x_min <= x <= self.x_max or \
                 not self.theta_min <= theta <= self.theta_max

        reward = self.reward(failed)

        self.times_at_goal += np.abs(s - self.goal_position) < self.goal_margin * self.world_width
        successful = self.times_at_goal >= self.goal_stable_duration

        done = failed or successful

        self.episode_step += 1

        return np.array(self.state), reward, done, {'success': successful}

    def render(self, mode='human'):

        s, s_dot, theta, theta_dot = self.state
        x, x_dot = self.x(s), self.x_dot(s)

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width_pixels, self.screen_height_pixels)

            # track / ground
            xs = np.linspace(self.x_min, self.x_max, 2000)
            # ys = np.array(self.y(xs))
            ys = np.array([self.y(t) for t in xs])
            xys = list(zip((xs - self.x_min) * self.scale, ys * self.scale))

            self.track = rendering.make_polyline([(0, 0), *xys, (self.screen_width_pixels, 0)])
            self.track.set_linewidth(5)
            self.viewer.add_geom(self.track)

            # cart
            l, r, t, b = -self.cart_width_pixels / 2, self.cart_width_pixels / 2, self.cart_height_pixels / 2, -self.cart_height_pixels / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_trans = rendering.Transform()
            cart.add_attr(rendering.Transform(translation=(0, self.wheels_radius + self.cart_height_pixels / 2)))
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # wheels
            front_wheel = rendering.make_circle(self.wheels_radius)
            front_wheel.set_color(0.5, 0.5, 0.5)
            front_wheel.add_attr(rendering.Transform(translation=(self.cart_width_pixels / 4, self.wheels_radius)))
            front_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(front_wheel)
            back_wheel = rendering.make_circle(self.cart_height_pixels / 3)
            back_wheel.set_color(0.5, 0.5, 0.5)
            back_wheel.add_attr(rendering.Transform(translation=(-self.cart_width_pixels / 4, self.wheels_radius)))
            back_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(back_wheel)

            # pole
            pole_line = LineString([(0, 0), (0, self.pole_length_pixels)]).buffer(self.pole_width_pixels / 2)
            pole = rendering.make_polygon(list(pole_line.exterior.coords))
            pole.set_color(0.8, 0.6, 0.4)
            self.pole_trans = rendering.Transform(translation=(0, self.cart_height_pixels + self.wheels_radius))
            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            self.viewer.add_geom(pole)

            # axle
            self.axle = rendering.make_circle(self.pole_width_pixels / 2)
            self.axle.add_attr(self.pole_trans)
            self.axle.add_attr(self.cart_trans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

            # flag
            flag_x = (self.goal_position - self.x_min) * self.scale
            flag_bottom_y = self.y(self.goal_position) * self.scale
            flag_top_y = flag_bottom_y + 100.0
            flagpole = rendering.Line((flag_x, flag_bottom_y), (flag_x, flag_top_y))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flag_x, flag_top_y), (flag_x, flag_top_y - 25), (flag_x + 50, flag_top_y - 15)])
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

            # goal margin
            stone_width, stone_height = 20, 40
            left_stone_x = (self.goal_position - self.world_width * self.goal_margin - self.x_min) * self.scale
            left_stone_bottom_y = self.y(left_stone_x / self.scale) * self.scale
            right_stone_x = (self.goal_position + self.world_width * self.goal_margin - self.x_min) * self.scale
            right_stone_bottom_y = self.y(right_stone_x / self.scale) * self.scale
            left_stone = rendering.FilledPolygon([(left_stone_x - stone_width / 2, left_stone_bottom_y),
                                                  (left_stone_x + stone_width / 2, left_stone_bottom_y),
                                                  (left_stone_x + stone_width / 2, left_stone_bottom_y + stone_height),
                                                  (left_stone_x - stone_width / 2, left_stone_bottom_y + stone_height)])
            right_stone = rendering.FilledPolygon([(right_stone_x - stone_width / 2, right_stone_bottom_y),
                                                   (right_stone_x + stone_width / 2, right_stone_bottom_y),
                                                   (right_stone_x + stone_width / 2,
                                                    right_stone_bottom_y + stone_height),
                                                   (right_stone_x - stone_width / 2,
                                                    right_stone_bottom_y + stone_height)])
            left_stone.set_color(255, 0, 0)
            right_stone.set_color(255, 0, 0)
            self.viewer.add_geom(left_stone)
            self.viewer.add_geom(right_stone)

        self.cart_trans.set_translation((x - self.x_min) * self.scale, self.y(x) * self.scale)
        k = np.arctan(-1 / self.y_dot(x)) if self.y_dot(x) != 0.0 else pi / 2
        self.cart_trans.set_rotation(pi / 2 + k if k < 0 else k - pi / 2)

        self.pole_trans.set_rotation(-(pi / 2 + k if k < 0 else k - pi / 2) - theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
