import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
from gym.envs.classic_control.cartpole_extension import CartPoleExtensionEnv

import numpy as np
from scipy.constants import g, pi
from scipy.integrate import odeint

from shapely.geometry import LineString


class CartPoleMountainEnv(CartPoleExtensionEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, mode='train', de_solver='scipy', seed=371875):
        self.name = 'CartPole-v3'

        self.world_width, self.world_height = 4 * pi, 2 * pi

        self.mode = mode

        self.gravity = -g
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = (self.mass_pole + self.mass_cart)
        self.pole_length = 1.0
        self.force_mag = 10
        self.tau = 0.02  # seconds between state updates

        self.de_solver = de_solver

        self.theta_min, self.theta_max = -pi / 2, pi / 2
        self.x_min, self.x_max = -self.world_width / 2, self.world_width / 2

        low = np.array([
            self.x_min * 2,
            -np.finfo(np.float32).max,
            self.theta_min * 2,
            -np.finfo(np.float32).max,
            0.0,
            -np.finfo(np.float32).max,
            -np.finfo(np.float32).max])
        high = np.array([
            self.x_max * 2,
            np.finfo(np.float32).max,
            self.theta_max * 2,
            np.finfo(np.float32).max,
            self.world_height, 
            np.finfo(np.float32).max, 
            np.finfo(np.float32).max])

        # self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.seed(seed)

        self.screen_width_pixels, self.screen_height_pixels = 1600, 800
        self.scale = self.screen_width_pixels / self.world_width

        self.t_evals = None

        self.cart_width = 0.4
        self.cart_height = 0.25
        self.track_height = 0.2

        self.cart_width_pixels = self.cart_width * self.scale
        self.cart_height_pixels = self.cart_height * self.scale
        self.track_height_pixels = self.track_height * self.scale
        self.wheels_radius = 0.3 * self.cart_height_pixels

        self.pole_width = 0.04
        self.pole_width_pixels = self.pole_width * self.scale
        self.pole_length_pixels = self.scale * self.pole_length

        # Environment parameters:
        self.initial_height = self.world_height / 3
        self.height = 1.5
        # self.height = 0.75
        self.steepness = 0.9

        self.slope_length = pi / self.steepness

        self.bottom = self.x_min + self.slope_length + pi / 2
        self.bottom_width = pi / 8

        self.offset = pi / (2 * self.steepness) + self.bottom

        self.starting_position = self.bottom
        self.goal_position = self.starting_position + 3 / 2 * pi

        self.viewer = None
        self.state = None

        self.episode_step = 0
        self.max_episode_steps = 500

        self.min_goal, self.max_goal = \
            self.bottom + self.bottom_width / 2 + self.slope_length, self.x_max
        self.goal_stable_duration = 10
        self.goal_x_dot_margin = 1.0
        self.goal_theta_margin = 0.05
        self.goal_theta_dot_margin = 1.0
        self.times_at_goal = 0

        self.R, self.r, self.d = 5.0 / 2, 3.0 / 2, 5.0 / 2

    def reset(self):

        # self.state = np.random.uniform(
        #     low=(self.bottom - self.bottom_width / 2, -0.05, -pi / 15, -0.05),
        #     high=(self.bottom + self.bottom_width / 2, 0.05, pi / 15, 0.05),
        #     size=(4,))

        if self.mode == 'train':
            self.state = np.random.uniform(
                low=(self.bottom - self.slope_length, -0.05, -pi / 12, -0.05),
                high=(self.goal_position + pi / 2, 0.05, pi / 12, 0.05),
                size=(4,))
        else:
            self.state = np.random.uniform(
                low=(self.bottom - self.bottom_width / 2,
                     -0.05, -pi / 15, -0.05),
                high=(self.bottom + self.bottom_width / 2,
                      0.05, pi / 15, 0.05),
                size=(4,))
        self.times_at_goal = 0
        self.episode_step = 0
        return self.obeservation()

    def x(self, s):
        return s

    def x_dot(self, s):
        return 1

    def x_dot_dot(self, s):
        return 0

    def y(self, s):
        w = np.array(s)
        w_left, w_right = w[w <= self.bottom], w[w > self.bottom]
        y_left = self.height * np.sin(
            self.steepness * (np.clip(w_left, self.bottom - self.bottom_width / 2 - self.slope_length,
                                      self.bottom - self.bottom_width / 2) - (self.offset - self.bottom_width / 2))) + self.initial_height
        y_right = self.height * np.sin(
            self.steepness * (np.clip(w_right, self.bottom + self.bottom_width / 2,
                                      self.bottom + self.bottom_width / 2 + self.slope_length) - (self.offset + self.bottom_width / 2))) + self.initial_height
        return np.concatenate((y_left, y_right))

    def y_dot(self, s):
        w = np.array(s)
        w_left, w_right = w[w <= self.bottom], w[w > self.bottom]
        y_left = self.height * self.steepness * np.cos(
            self.steepness * (w_left - (self.offset - self.bottom_width / 2)))
        y_right = self.height * self.steepness * np.cos(
            self.steepness * (w_right - (self.offset + self.bottom_width / 2)))
        y_left[w_left <= self.bottom -
               self.bottom_width / 2 - self.slope_length] = 0.0
        y_left[w_left >= self.bottom - self.bottom_width / 2] = 0.0
        y_right[w_right <= self.bottom + self.bottom_width / 2] = 0.0
        y_right[w_right >= self.bottom +
                self.bottom_width / 2 + self.slope_length] = 0.0
        return np.concatenate((y_left, y_right))

    def y_dot_dot(self, s):
        w = np.array(s)
        w_left, w_right = w[w <= self.bottom], w[w > self.bottom]
        y_left = -self.height * self.steepness**2 * \
            np.sin(self.steepness * (w_left - (self.offset - self.bottom_width / 2)))
        y_right = -self.height * self.steepness**2 * \
            np.sin(self.steepness * (w_right -
                                     (self.offset + self.bottom_width / 2)))
        y_left[w_left <= self.bottom -
               self.bottom_width / 2 - self.slope_length] = 0.0
        y_left[w_left >= self.bottom - self.bottom_width / 2] = 0.0
        y_right[w_right <= self.bottom + self.bottom_width / 2] = 0.0
        y_right[w_right >= self.bottom +
                self.bottom_width / 2 + self.slope_length] = 0.0
        return np.concatenate((y_left, y_right))

    """
    def x(self, s):
        return (self.R - self.r) * np.cos(s) + self.d * np.cos((self.R - self.r) / self.r * s)

    def x_dot(self, s):
        return -np.sin(s)

    def x_dot_dot(self, s):
        return -np.cos(s)

    def y(self, s):
        return (self.R - self.r) * np.cos(s) - self.d * np.sin((self.R - self.r) / self.r * s) + self.initial_height

    def y_dot(self, s):
        return np.cos(s)
        
    def y_dot_dot(self, s):
        return -np.sin(s)

    def x(self, s):
        return np.cos(s)

    def x_dot(self, s):
        return -np.sin(s)

    def x_dot_dot(self, s):
        return -np.cos(s)

    def y(self, s):
        return np.sin(s) + self.initial_height

    def y_dot(self, s):
        return np.cos(s)
        
    def y_dot_dot(self, s):
        return -np.sin(s)
    """

    def action_to_force(self, action):
        return (action - 1) * self.force_mag

    def reward(self, in_goal_state, failed):
        if in_goal_state:
            return 0
        elif failed:
            return -0.5
        else:
            return -1 / (2 * self.max_episode_steps)

    def in_goal_state(self):
        x = self.x(self.state[0])
        return self.theta_min <= self.state[2] <= self.theta_max and \
            self.min_goal <= x <= self.max_goal

    def is_successful(self, in_goal_state, failed):

        if self.mode == 'train':
            return self.times_at_goal >= self.goal_stable_duration

        elif self.mode in ['test', 'eval']:

            return not failed and \
                in_goal_state and \
                self.episode_step >= self.max_episode_steps - 1

    def has_failed(self, x, theta):
        return not self.x_min <= x <= self.x_max or \
            not self.theta_min <= theta <= self.theta_max or \
            (self.episode_step >= self.max_episode_steps - 1 and 
             not self.in_goal_state())

    def obeservation(self):
        s, s_dot, theta, theta_dot = self.state
        return np.array([s, s_dot, theta, theta_dot,
                         self.y(s)[0], self.y_dot(s)[0], self.y_dot_dot(s)[0]])

    def render(self, mode='human'):

        s, s_dot, theta, theta_dot = self.state
        x, x_dot = self.x(s), self.x_dot(s)

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width_pixels,
                                           self.screen_height_pixels)

            # track / ground
            ss = np.linspace(self.x_min, self.x_max, 2000)
            xs = np.array(self.x(ss))
            ys = np.array(self.y(ss))
            xys = list(zip((xs - self.x_min) * self.scale, ys * self.scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_color(44/255, 160/255, 44/255)
            self.track.set_linewidth(5)
            self.viewer.add_geom(self.track)

            # start flag
            flag_x = (self.x(self.starting_position) - self.x_min) * self.scale
            flag_bottom_y = self.y(self.starting_position) * self.scale
            flag_top_y = flag_bottom_y + 100.0
            flagpole = rendering.Line((flag_x, flag_bottom_y),
                                      (flag_x, flag_top_y))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([
                (flag_x, flag_top_y),
                (flag_x, flag_top_y - 25),
                (flag_x + 50, flag_top_y - 15)])
            flag.set_color(105/255, 183/255, 100/255)
            self.viewer.add_geom(flag)

            # goal flag
            flag_x = (self.x(self.goal_position) - self.x_min) * self.scale
            flag_bottom_y = self.y(self.goal_position) * self.scale
            flag_top_y = flag_bottom_y + 100.0
            flagpole = rendering.Line((flag_x, flag_bottom_y),
                                      (flag_x, flag_top_y))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([
                (flag_x, flag_top_y),
                (flag_x, flag_top_y - 25),
                (flag_x + 50, flag_top_y - 15)])
            flag.set_color(255/255, 221/255, 113/255)
            self.viewer.add_geom(flag)

            # goal margin
            stone_width, stone_height = 16, 16
            left_stone_x = (self.x(self.min_goal) - self.x_min) * self.scale
            left_stone_bottom_y = \
                self.y(left_stone_x / self.scale) * self.scale
            right_stone_x = (self.x(self.max_goal) - self.x_min) * self.scale
            right_stone_bottom_y = \
                self.y(right_stone_x / self.scale) * self.scale
            left_stone = rendering.FilledPolygon([
                (left_stone_x, left_stone_bottom_y),
                (left_stone_x + stone_width, left_stone_bottom_y),
                (left_stone_x + stone_width / 2,
                 left_stone_bottom_y + stone_height)])
            right_stone = rendering.FilledPolygon([
                (right_stone_x - stone_width, right_stone_bottom_y),
                (right_stone_x, right_stone_bottom_y),
                (right_stone_x - stone_width / 2,
                 right_stone_bottom_y + stone_height)])
            left_stone.set_color(237/255, 102/255, 93/255)
            right_stone.set_color(237/255, 102/255, 93/255)
            self.viewer.add_geom(left_stone)
            self.viewer.add_geom(right_stone)

            for ii in range(0, 8 + 1):
                marker = rendering.FilledPolygon([
                    (ii * pi / 2 * self.scale - stone_width / 4, 0),
                    (ii * pi / 2 * self.scale - stone_width / 4,
                     stone_height / 2),
                    (ii * pi / 2 * self.scale + stone_width / 4,
                     stone_height / 2),
                    (ii * pi / 2 * self.scale + stone_width / 4, 0)])
                marker.set_color(242/255, 108/255, 100/255)
                self.viewer.add_geom(marker)

            marker = rendering.FilledPolygon([
                ((self.bottom - self.x_min) * self.scale - stone_width / 4, 0),
                ((self.bottom - self.x_min) * self.scale - stone_width / 4,
                 stone_height / 2),
                ((self.bottom - self.x_min) * self.scale + stone_width / 4,
                 stone_height / 2),
                ((self.bottom - self.x_min) * self.scale + stone_width / 4,
                 0)])
            marker.set_color(255/255, 193/255, 86/255)
            self.viewer.add_geom(marker)

            # cart
            l, r, t, b = [-self.cart_width_pixels / 2,
                          self.cart_width_pixels / 2,
                          self.cart_height_pixels / 2,
                          -self.cart_height_pixels / 2]
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cart.set_color(96/255, 99/255, 106/255)
            self.cart_trans = rendering.Transform()
            cart.add_attr(rendering.Transform(
                translation=(0, self.wheels_radius +
                             self.cart_height_pixels / 2)))
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # wheels
            front_wheel = rendering.make_circle(self.wheels_radius)
            front_wheel.set_color(65/255, 68/255, 81/255)
            front_wheel.add_attr(rendering.Transform(
                translation=(self.cart_width_pixels / 4, self.wheels_radius)))
            front_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(front_wheel)
            back_wheel = rendering.make_circle(self.wheels_radius)
            back_wheel.set_color(65/255, 68/255, 81/255)
            back_wheel.add_attr(rendering.Transform(
                translation=(-self.cart_width_pixels / 4, self.wheels_radius)))
            back_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(back_wheel)

            # pole
            pole_line = LineString([
                (0, 0),
                (0, self.pole_length_pixels)]
            ).buffer(self.pole_width_pixels / 2)
            pole = rendering.make_polygon(list(pole_line.exterior.coords))
            pole.set_color(168/255, 120/255, 110/255)
            self.pole_trans = rendering.Transform(
                translation=(0, self.cart_height_pixels + self.wheels_radius))
            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            self.viewer.add_geom(pole)

            # axle
            self.axle = rendering.make_circle(self.pole_width_pixels / 2)
            self.axle.add_attr(self.pole_trans)
            self.axle.add_attr(self.cart_trans)
            self.axle.set_color(127/255, 127/255, 127/255)
            self.viewer.add_geom(self.axle)

        self.cart_trans.set_translation(
            (x - self.x_min) * self.scale, self.y(s) * self.scale)
        k = np.arctan(-1 / self.y_dot(x)) if self.y_dot(x) != 0.0 else pi / 2
        self.cart_trans.set_rotation(pi / 2 + k if k < 0 else k - pi / 2)

        self.pole_trans.set_rotation(
            -(pi / 2 + k if k < 0 else k - pi / 2) - theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
