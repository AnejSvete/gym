import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

from gym.envs.classic_control.cartpole_extension import CartPoleExtensionEnv

import numpy as np
from scipy.constants import g, pi
from scipy.integrate import solve_ivp

from shapely.geometry import LineString, Polygon

# TODO: as strings...
PUSH_CART_RIGHT = 0
PUSH_CART_LEFT = 1
PUSH_POLE_RIGHT = 2
PUSH_POLE_LEFT = 3
STOP_CART = 4
STOP_POLE = 5


class CartPoleAltEnv(CartPoleExtensionEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, mode='train', de_solver='scipy', seed=526245):

        self.world_width, self.world_height = 5 * pi, pi

        self.gravity = -g
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_cart
        self.pole_length = 1.0
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.de_solver = de_solver
        self.mode = mode

        self.theta_min, self.theta_max = -pi / 4, pi / 4
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

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.seed(seed)

        self.screen_width_pixels, self.screen_height_pixels = 2000, 400
        self.scale = self.screen_width_pixels / self.world_width

        self.t_evals = None 
        
        self.cart_width = 0.4
        self.cart_height = 0.25
        self.track_height = 0.2

        self.cart_width_pixels = self.cart_width * self.scale
        self.cart_height_pixels = self.cart_height * self.scale
        self.track_height_pixels = self.track_height * self.scale
        self.wheels_radius = 0.3 * self.cart_height_pixels
        self.cart_middle_y_pixels = self.track_height_pixels + \
            self.wheels_radius + \
            self.cart_height_pixels / 2
        self.cart_top_y_pixels = self.cart_middle_y_pixels + \
            self.cart_height_pixels / 2
        self.cart_bottom_y_pixels = self.cart_middle_y_pixels - \
            self.cart_height_pixels / 2

        self.cart_middle_y = self.cart_middle_y_pixels / self.scale
        self.cart_top_y = self.cart_top_y_pixels / self.scale
        self.cart_bottom_y = self.cart_bottom_y_pixels / self.scale

        self.pole_width = 0.04
        self.pole_width_pixels = self.pole_width * self.scale
        self.pole_length_pixels = self.pole_length * self.scale

        self.pole_bottom_y_pixels = self.cart_top_y_pixels
        self.pole_bottom_y = self.pole_bottom_y_pixels / self.scale

        self.viewer = None
        self.state = None

        self.episode_step = 0
        self.max_episode_steps = 500

        self.goal_stable_duration = 500
        
    def reset(self):

        self.state = self.np_random.uniform(
            low=(-0.05, -0.05, -pi / 12, -0.05),
            high=(0.05, 0.05, pi / 12, 0.05),
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
        return self.track_height

    def y_dot(self, s):
        return 0.0

    def y_dot_dot(self, s):
        return 0.0

    def action_to_force(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def reward(self, in_goal_state, failed):
        return 0 if failed else 1

    def in_goal_state(self):
        s, _, theta, _ = self.state
        x = self.x(s)
        return not self.has_failed(x, theta)

    def has_failed(self, x, theta):
        return not self.x_min <= x <= self.x_max or \
            not self.theta_min <= theta <= self.theta_max 

    def obeservation(self):
        return self.state

    def render(self, mode='human'):

        x, x_dot, theta, theta_dot = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width_pixels,
                                           self.screen_height_pixels)

            # track / ground
            self.track = rendering.FilledPolygon([
                (0, 0),
                (0, self.track_height_pixels),
                (self.screen_width_pixels, self.track_height_pixels),
                (self.screen_width_pixels, 0)])
            self.track.set_color(44/255, 160/255, 44/255)
            self.viewer.add_geom(self.track)

            # cart
            l, r, t, b = [-self.cart_width_pixels / 2,
                          self.cart_width_pixels / 2,
                          self.cart_height_pixels / 2,
                          -self.cart_height_pixels / 2]
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cart.set_color(96/255, 99/255, 106/255)
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # wheels
            front_wheel = rendering.make_circle(self.wheels_radius)
            front_wheel.set_color(65/255, 68/255, 81/255)
            front_wheel.add_attr(rendering.Transform(
                translation=(self.cart_width_pixels / 4,
                             -self.cart_height_pixels / 2)))
            front_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(front_wheel)
            back_wheel = rendering.make_circle(self.wheels_radius)
            back_wheel.set_color(65/255, 68/255, 81/255)
            back_wheel.add_attr(rendering.Transform(
                translation=(-self.cart_width_pixels / 4,
                             -self.cart_height_pixels / 2)))
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
                translation=(0, self.cart_height_pixels / 2))
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
            x * self.scale + self.screen_width_pixels / 2,
            self.cart_middle_y_pixels)

        self.pole_trans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
