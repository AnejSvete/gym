import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
from scipy.constants import g, pi
from scipy.integrate import odeint

from shapely.geometry import LineString, Polygon

# TODO
PUSH_CART_RIGHT = 0
PUSH_CART_LEFT = 1
PUSH_POLE_RIGHT = 2
PUSH_POLE_LEFT = 3
STOP_CART = 4
STOP_POLE = 5


class CartPoleSimpleObstacleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self, mode='train', de_solver='scipy'):

        self.seed()

        self.world_width, self.world_height = 3 * pi, pi

        self.gravity = -g
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = (self.mass_pole + self.mass_cart)
        self.length = 0.5  # actually half the pole's length
        self.pole_mass_length = (self.mass_pole * self.length)
        self.force_mag = 10.0
        self.tau = 0.01  # seconds between state updates

        self.de_solver = de_solver
        self.mode = mode

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

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.screen_width_pixels, self.screen_height_pixels = 1800, 600
        self.scale = self.screen_width_pixels / self.world_width

        self.cart_width_pixels = 100.0
        self.cart_height_pixels = 60.0
        self.track_height_pixels = 50.0
        self.wheels_radius = self.cart_height_pixels / 3
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
        self.cart_width = self.cart_width_pixels / self.scale
        self.cart_height = self.cart_height_pixels / self.scale
        self.track_height = self.track_height_pixels / self.scale

        self.pole_width_pixels = 8
        self.pole_length_pixels = self.scale * (2 * self.length)

        self.pole_bottom_y_pixels = self.cart_top_y_pixels
        self.pole_bottom_y = self.pole_bottom_y_pixels / self.scale

        self.obstacle_width_pixels = 8
        self.obstacle_height_pixels = 0.4 * self.screen_height_pixels
        self.obstacle_location = self.x_min + self.world_width / 3
        self.obstacle_location_pixels = \
            (self.obstacle_location - self.x_min) * self.scale
        self.obstacle_coordinate_pixels = [
            self.obstacle_location_pixels - self.obstacle_width_pixels / 2,
            self.obstacle_location_pixels + self.obstacle_width_pixels / 2,
            self.screen_height_pixels,
            self.screen_height_pixels - self.obstacle_height_pixels]

        self.pole_length = self.pole_length_pixels / self.scale

        self.starting_position = self.obstacle_location - pi / 2
        self.goal_position = self.starting_position + 3 / 2 * pi

        self.intersection_polygon = None

        self.viewer = None
        self.state = None

        self.episode_step = 0
        self.max_episode_steps = 1000

        self.goal_margin = pi / 3
        self.goal_stable_duration = 200
        self.times_at_goal = 0

    def reset(self, epoch=-1, num_epochs=-1):
        if self.mode == 'train':
            self.state = np.random.uniform(
                low=(self.starting_position - pi / 4,
                     -0.05, -pi / 6, -0.05),
                high=(self.goal_position + pi / 2,
                      0.05, pi / 6, 0.05),
                size=(4,))
        else:
            self.state = np.random.uniform(
                low=(self.starting_position, -0.05, -pi / 15, -0.05),
                high=(self.starting_position, 0.05, pi / 15, 0.05),
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

    def x_dot_dot(self, force, theta, theta_dot, theta_dot_dot):
        temp = (force + self.pole_mass_length * theta_dot * theta_dot * np.sin(theta)) / self.total_mass
        return temp - self.pole_mass_length * theta_dot_dot * np.cos(theta) / self.total_mass

    def theta_dot_dot(self, force, theta, theta_dot):
        temp = (force + self.pole_mass_length * theta_dot * theta_dot * np.sin(theta)) / self.total_mass
        return (self.gravity * np.sin(theta) - np.cos(theta)* temp) / (self.length * (4.0/3.0 - self.mass_pole * np.cos(theta) * np.cos(theta) / self.total_mass))

    def pole_top_coordinates(self, screen_coordinates=True):
        x, x_dot, theta, theta_dot = self.state
        if screen_coordinates:
            return (x * self.scale + self.screen_width_pixels / 2 +
                    self.pole_length_pixels * np.sin(theta),
                    self.cart_top_y_pixels +
                    self.pole_length_pixels * np.cos(theta))
        else:
            return (x + self.pole_length * np.sin(theta),
                    self.track_height + self.cart_height +
                    self.pole_length * np.cos(theta))

    def pole_bottom_coordinates(self, screen_coordinates=True):
        x, x_dot, theta, theta_dot = self.state
        if screen_coordinates:
            return x * self.scale + self.screen_width_pixels / 2, \
                   self.pole_bottom_y_pixels
        else:
            return x, self.pole_bottom_y

    def pole_touches_obstacle(self):
        if self.state is None:
            return True

        l, r, t, b = self.obstacle_coordinate_pixels
        obstacle = Polygon([(l, b), (l, t), (r, t), (r, b)])
        pole = LineString([self.pole_bottom_coordinates(),
                           self.pole_top_coordinates()]).buffer(
            self.pole_width_pixels / 2)
        intersection = obstacle.intersection(pole)

        if intersection.is_empty:
            self.intersection_polygon = None
            return False
        else:
            self.intersection_polygon = intersection
            return True

    def new_state(self, action):

        x, x_dot, theta, theta_dot = self.state

        if action in [-1, -2]:
            force = 10 * self.force_mag \
                if action == -1 else -10 * self.force_mag
        else:
            force = self.force_mag if action == 1 else -self.force_mag

        if self.de_solver == 'euler':

            theta_dot_dot = self.theta_dot_dot(force, theta, theta_dot)
            x_dot_dot = self.x_dot_dot(force, theta, theta_dot, theta_dot_dot)

            x += self.tau * x_dot
            x_dot += self.tau * x_dot_dot
            theta += self.tau * theta_dot
            theta_dot += self.tau * theta_dot_dot

        elif self.de_solver == 'scipy':
            def dx(z, t, force=0.0):
                self.x_dot, self.x = z
                return np.array((self.x_dot_dot(
                    force, theta, theta_dot, theta_dot_dot), z[0]))

            def dtheta(z, t, force=0.0):
                self.theta_dot, self.theta = z
                return np.array((self.theta_dot_dot(
                    force, theta, theta_dot), z[0]))

            t = np.linspace(0, self.tau, num=2)

            theta_dot_tmp, theta_tmp = odeint(dtheta,
                                              np.array([theta_dot, theta]),
                                              t, args=(force,)).T

            x_dot_tmp, x_tmp = odeint(dx, np.array([x_dot, x]), t,
                                      args=(force,)).T

            x_dot = x_dot_tmp[-1]
            x = x_tmp[-1]

            theta_dot = theta_dot_tmp[-1]
            theta = theta_tmp[-1]

        if theta < -pi:
            theta += 2 * pi
        elif theta > pi:
            theta -= 2 * pi

        return np.array([x, x_dot, theta, theta_dot])

    def reward(self, failed):
        x, x_dot, theta, theta_dot = self.state
        if failed:
            # return -2 * (self.max_episode_steps - self.episode_step)
            # return -2 * (self.max_episode_steps - self.episode_step) / \
            #        (2 * self.max_episode_steps)
            return -0.5
        else:
            return 0 if np.abs(x - self.goal_position) < self.goal_margin \
                else -1 / (2 * self.max_episode_steps)
            # return 0 if np.abs(x - self.goal_position) < self.goal_margin \
            #     else -1

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        x, x_dot, theta, theta_dot = self.new_state(action)
        self.state = (x, x_dot, theta, theta_dot)
        failed = not self.x_min <= x <= self.x_max or \
                 not self.theta_min <= theta <= self.theta_max or \
                 self.pole_touches_obstacle() or \
                 self.episode_step >= self.max_episode_steps - 1

        reward = self.reward(failed)

        if np.abs(x - self.goal_position) < self.goal_margin:
            self.times_at_goal += 1
        else:
            self.times_at_goal = 0

        successful = self.times_at_goal >= self.goal_stable_duration

        done = failed or successful

        self.episode_step += 1

        info = {'success': successful,
                'time_limit': self.episode_step >= self.max_episode_steps}

        return np.array(self.state), reward, done, info

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
            self.track.set_color(0, 255, 0)
            self.viewer.add_geom(self.track)

            # flag
            flag_x = (self.goal_position - self.x_min) * self.scale
            flag_bottom_y = self.track_height_pixels
            flag_top_y = flag_bottom_y + 100.0
            flagpole = rendering.Line((flag_x, flag_bottom_y),
                                      (flag_x, flag_top_y))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flag_x, flag_top_y),
                                            (flag_x, flag_top_y - 30),
                                            (flag_x + 50, flag_top_y - 15)])
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

            # goal margin
            stone_width, stone_height = 20, 20
            stone_bottom_y = self.track_height_pixels
            left_stone_x = (self.goal_position -
                            self.goal_margin - self.x_min) * self.scale
            right_stone_x = (self.goal_position +
                             self.goal_margin - self.x_min) * self.scale
            left_stone = rendering.FilledPolygon([
                (left_stone_x, stone_bottom_y),
                (left_stone_x + stone_width, stone_bottom_y),
                (left_stone_x + stone_width / 2,
                 stone_bottom_y + stone_height)])
            right_stone = rendering.FilledPolygon([
                (right_stone_x - stone_width, stone_bottom_y),
                (right_stone_x, stone_bottom_y),
                (right_stone_x - stone_width / 2,
                 stone_bottom_y + stone_height)])
            left_stone.set_color(0, 255, 0)
            right_stone.set_color(0, 255, 0)
            self.viewer.add_geom(left_stone)
            self.viewer.add_geom(right_stone)

            # cart
            l, r, t, b = [-self.cart_width_pixels / 2,
                          self.cart_width_pixels / 2,
                          self.cart_height_pixels / 2,
                          -self.cart_height_pixels / 2]
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # wheels
            front_wheel = rendering.make_circle(self.wheels_radius)
            front_wheel.set_color(0.5, 0.5, 0.5)
            front_wheel.add_attr(rendering.Transform(
                translation=(self.cart_width_pixels / 4,
                             -self.cart_height_pixels / 2)))
            front_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(front_wheel)
            back_wheel = rendering.make_circle(self.cart_height_pixels / 3)
            back_wheel.set_color(0.5, 0.5, 0.5)
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
            pole.set_color(0.8, 0.6, 0.4)
            self.pole_trans = rendering.Transform(
                translation=(0, self.cart_height_pixels / 2))
            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            self.viewer.add_geom(pole)

            # axle
            self.axle = rendering.make_circle(self.pole_width_pixels / 2)
            self.axle.add_attr(self.pole_trans)
            self.axle.add_attr(self.cart_trans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

            # obstacle
            l, r, t, b = self.obstacle_coordinate_pixels
            obstacle = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            obstacle.set_color(255, 0, 0)
            self.viewer.add_geom(obstacle)

        if self.intersection_polygon is not None:
            intersection_polygon = rendering.FilledPolygon(
                list(self.intersection_polygon.exterior.coords))
            intersection_polygon.set_color(0.75, 0.75, 0.75)
            self.viewer.add_onetime(intersection_polygon)

        self.cart_trans.set_translation(
            x * self.scale + self.screen_width_pixels / 2,
            self.cart_middle_y_pixels)

        self.pole_trans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
