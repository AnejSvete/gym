import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
from scipy.constants import g, pi
from scipy.integrate import odeint

from shapely.geometry import LineString, Polygon


class CartPoleObstacleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.world_width, self.world_height = 4 * pi, 2 * pi

        self.gravity = -g
        self.mass_cart = 10.0
        self.mass_pole = 1.0
        self.total_mass = (self.mass_pole + self.mass_cart)
        self.length = 1.0  # actually half the pole's length
        self.pole_mass_length = (self.mass_pole * self.length)
        self.force_mag = 50.0
        self.tau = 0.02  # seconds between state updates

        self.de_solver = 'euler'

        self.theta_min, self.theta_max = -pi / 4, pi / 4
        self.x_min, self.x_max = -self.world_width / 2, self.world_width / 2

        low = np.array([self.x_min * 2, -np.finfo(np.float32).max, self.theta_min * 2, -np.finfo(np.float32).max])
        high = np.array([self.x_max * 2, np.finfo(np.float32).max, self.theta_max * 2, np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.screen_width_pixels, self.screen_height_pixels = 1600, 800
        self.scale = self.screen_width_pixels / self.world_width

        self.cart_width_pixels = 100.0
        self.cart_height_pixels = 60.0
        self.track_height_pixels = 50.0
        self.wheels_radius = self.cart_height_pixels / 3
        self.cart_middle_y_pixels = self.track_height_pixels + self.wheels_radius + self.cart_height_pixels / 2
        self.cart_top_y_pixels = self.cart_middle_y_pixels + self.cart_height_pixels / 2
        self.cart_bottom_y_pixels = self.cart_middle_y_pixels - self.cart_height_pixels / 2

        self.cart_middle_y = self.cart_middle_y_pixels / self.scale
        self.cart_top_y = self.cart_top_y_pixels / self.scale
        self.cart_bottom_y = self.cart_bottom_y_pixels / self.scale
        self.cart_width = self.cart_width_pixels / self.scale
        self.cart_height = self.cart_height_pixels / self.scale
        self.track_height = self.track_height_pixels / self.scale

        self.pole_width_pixels = 20.0
        self.pole_length_pixels = self.scale * (2 * self.length)

        self.pole_bottom_y_pixels = self.cart_top_y_pixels
        self.pole_bottom_y = self.pole_bottom_y_pixels / self.scale

        self.obstacle_width_pixels, self.obstacle_height_pixels = self.screen_width_pixels / 5, self.screen_height_pixels / 5
        self.obstacle_coordinate_pixels = [self.screen_width_pixels / 2 - self.obstacle_width_pixels / 2,
                                           self.screen_width_pixels / 2 + self.obstacle_width_pixels / 2,
                                           self.cart_top_y_pixels + self.pole_length_pixels + self.obstacle_height_pixels / 2 + 150,
                                           self.cart_top_y_pixels + self.pole_length_pixels - self.obstacle_height_pixels / 2 + 150]

        self.pole_length = self.pole_length_pixels / self.scale

        self.goal_position = 3 / 2 * pi

        self.intersection_polygon = None

        self.seed()
        self.viewer = None
        self.previous_state, self.state = None, None

        self.times_at_goal = 0

    def reset(self):
        self.state = self.np_random.uniform(low=(-3 * pi / 2, -0.05, -0.05, -0.05),
                                            high=(-pi, 0.05, 0.05, 0.05),
                                            size=(4,))
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

    def phi(self, t):
        return np.arctan(self.y_dot(t))

    def s_dot_dot(self, force=0.0):
        s, s_dot, theta, theta_dot = self.state
        return (-2 * force + self.y_dot(s) * (-(self.gravity * (self.mass_pole + 2 * self.mass_cart +
                                                                self.mass_pole * np.cos(2*theta))) -
                                              2 * self.pole_length * self.mass_pole * np.cos(theta) * theta_dot**2 +
                                              s_dot**2 * (self.mass_pole * np.sin(2 * theta) * self.x_dot_dot(s) +
                                                          (self.mass_pole + 2 * self.mass_cart +
                                                           self.mass_pole * np.cos(2 * theta)) * self.y_dot_dot(s))) +
                self.x_dot(s) * (-(self.gravity * self.mass_pole * np.sin(2*theta)) -
                                 2 * self.pole_length * self.mass_pole * np.sin(theta) * theta_dot**2 +
                                 s_dot**2 * ((self.mass_pole + 2 * self.mass_cart -
                                              self.mass_pole * np.cos(2*theta)) * self.x_dot_dot(s) +
                                             self.mass_pole * np.sin(2 * theta) * self.y_dot_dot(s)))) / \
               ((-self.mass_pole - 2 * self.mass_cart + self.mass_pole * np.cos(2*theta)) * self.x_dot(s)**2 -
                2 * self.mass_pole * np.sin(2 * theta) * self.x_dot(s) * self.y_dot(s) -
                (self.mass_pole + 2*self.mass_cart + self.mass_pole*np.cos(2*theta))*self.y_dot(s)**2)

    def theta_dot_dot(self, force=0.0):
        s, s_dot, theta, theta_dot = self.state
        return (force * (np.cos(theta) * self.x_dot(s) - np.sin(theta) * self.y_dot(s)) +
                (np.sin(theta) * self.x_dot(s) + np.cos(theta) * self.y_dot(s)) *
                (self.y_dot(s) * (-(self.pole_length * self.mass_pole * np.sin(theta) * theta_dot**2) +
                                  (self.mass_pole + self.mass_cart) * s_dot**2 * self.x_dot_dot(s)) +
                 self.x_dot(s) * (self.pole_length * self.mass_pole * np.cos(theta) * theta_dot**2 +
                                  (self.mass_pole + self.mass_cart) *
                                  (self.gravity - s_dot**2 * self.y_dot_dot(s))))) / \
               (self.pole_length * ((-self.mass_pole - self.mass_cart +
                                     self.mass_pole * np.cos(theta)**2) * self.x_dot(s)**2 -
                                    self.mass_pole * np.sin(2 * theta) * self.x_dot(s) * self.y_dot(s) -
                                    ((self.mass_pole + 2 * self.mass_cart +
                                      self.mass_pole * np.cos(2 * theta)) * self.y_dot(s)**2) / 2))

    def pole_top_coordinates(self, screen_coordinates=True):
        s, s_dot, theta, theta_dot = self.state
        x = self.x(s)
        if screen_coordinates:
            return (x * self.scale + self.screen_width_pixels / 2 + self.pole_length_pixels * np.sin(theta),
                    self.cart_top_y_pixels + self.pole_length_pixels * np.cos(theta))
        else:
            return (x + self.pole_length * np.sin(theta),
                    self.track_height + self.cart_height + self.pole_length * np.cos(theta))

    def pole_bottom_coordinates(self, screen_coordinates=True):
        s, s_dot, theta, theta_dot = self.state
        x = self.x(s)
        if screen_coordinates:
            return x * self.scale + self.screen_width_pixels / 2, self.pole_bottom_y_pixels
        else:
            return x, self.pole_bottom_y

    def pole_touches_obstacle(self):

        l, r, t, b = self.obstacle_coordinate_pixels
        obstacle = Polygon([(l, b), (l, t), (r, t), (r, b)])
        pole = LineString([self.pole_bottom_coordinates(), self.pole_top_coordinates()]).buffer(self.pole_width_pixels / 2)
        intersection = obstacle.intersection(pole)

        if intersection.is_empty:
            return False
        else:
            self.intersection_polygon = intersection
            return True

    def new_state(self, action):

        s, s_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag

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
                print(self.theta_dot, self.theta)
                return np.array((self.theta_dot_dot(force), z[0]))

            force = self.force_mag * (int(action) - 1)

            sample_length = 100
            t = np.linspace(0, 1, num=sample_length)

            s_dot_tmp, s_tmp = odeint(ds, np.array([s_dot, s]), t, args=(force,)).T
            theta_dot_tmp, theta_tmp = odeint(dtheta, np.array([theta_dot, theta]), t, args=(force,)).T

            s_dot = s_dot_tmp[int(sample_length / 50)]
            s = s_tmp[int(sample_length / 50)]

            theta_dot = theta_dot_tmp[int(sample_length / 50)]
            theta = theta_tmp[int(sample_length / 50)]

        if theta < -pi:
            theta += 2 * pi
        elif theta > pi:
            theta -= 2 * pi

        return np.array([s, s_dot, theta, theta_dot])

    def reward(self, done):
        current_x, _, _, _ = self.state
        current_distance_from_goal = np.abs(current_x - self.goal_position)
        # previous_x, _, _, _ = self.previous_state
        # previous_distance_from_goal = np.abs(previous_x - self.goal_position_world)
        # distance_difference = previous_distance_from_goal - current_distance_from_goal
        # return np.exp2(distance_difference) if current_distance_from_goal < 0.1 * self.world_width else np.exp2(-current_distance_from_goal)
        # return distance_difference if current_distance_from_goal < 0.1 * self.world_width else np.power(0.5, -current_distance_from_goal)
        # return self.world_width / (current_distance_from_goal + 1)
        return self.times_at_goal if current_distance_from_goal < 0.1 * self.world_width else -1 if done else 0.0

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        tmp_state = self.state
        x, x_dot, theta, theta_dot = self.new_state(action)
        self.state = (x, x_dot, theta, theta_dot)
        self.previous_state = tmp_state

        if self.pole_touches_obstacle():
            return np.array(self.state), -100, True, {}

        distance_from_goal = np.abs(x - self.goal_position)

        done = not self.x_min <= x <= self.x_max or \
               not self.theta_min <= theta <= self.theta_max or \
               self.times_at_goal >= 50

        if distance_from_goal < 0.1 * self.world_width:
            self.times_at_goal += 1
        else:
            self.times_at_goal = 0

        reward = self.reward(done)

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):

        x, x_dot, theta, theta_dot = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width_pixels, self.screen_height_pixels)

            # track / ground
            self.track = rendering.FilledPolygon([(0, 0),
                                                  (0, self.track_height_pixels),
                                                  (self.screen_width_pixels, self.track_height_pixels),
                                                  (self.screen_width_pixels, 0)])
            self.track.set_color(0, 255, 0)
            self.viewer.add_geom(self.track)

            # cart
            l, r, t, b = -self.cart_width_pixels / 2, self.cart_width_pixels / 2, self.cart_height_pixels / 2, -self.cart_height_pixels / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # wheels
            front_wheel = rendering.make_circle(self.wheels_radius)
            front_wheel.set_color(0.5, 0.5, 0.5)
            front_wheel.add_attr(rendering.Transform(translation=(self.cart_width_pixels / 4, -self.cart_height_pixels / 2)))
            front_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(front_wheel)
            back_wheel = rendering.make_circle(self.cart_height_pixels / 3)
            back_wheel.set_color(0.5, 0.5, 0.5)
            back_wheel.add_attr(rendering.Transform(translation=(-self.cart_width_pixels / 4, -self.cart_height_pixels / 2)))
            back_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(back_wheel)

            # pole
            pole_line = LineString([(0, 0), (0, self.pole_length_pixels)]).buffer(self.pole_width_pixels / 2)
            pole = rendering.make_polygon(list(pole_line.exterior.coords))
            pole.set_color(0.8, 0.6, 0.4)
            self.pole_trans = rendering.Transform(translation=(0, self.cart_height_pixels / 2))
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
            flag_bottom_y = self.track_height_pixels
            flag_top_y = flag_bottom_y + 200.0
            flagpole = rendering.Line((flag_x, flag_bottom_y), (flag_x, flag_top_y))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flag_x, flag_top_y), (flag_x, flag_top_y - 50), (flag_x + 100, flag_top_y - 30)])
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

            # obstacle
            l, r, t, b = self.obstacle_coordinate_pixels
            obstacle = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            obstacle.set_color(255, 0, 0)
            self.viewer.add_geom(obstacle)

        if self.intersection_polygon is not None:
            intersection_polygon = rendering.FilledPolygon(list(self.intersection_polygon.exterior.coords))
            intersection_polygon.set_color(0.75, 0.75, 0.75)
            self.viewer.add_onetime(intersection_polygon)

        self.cart_trans.set_translation(x * self.scale + self.screen_width_pixels / 2, self.cart_middle_y_pixels)

        self.pole_trans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
