
"""
InitialPath is the class for generating the naive initial path for NeuPAN from the given waypoints.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from math import tan, inf, cos, sin, sqrt
from gctl import curve_generator
from neupan.util import WrapToPi, distance
import math

class InitialPath:
    """
    generate initial path from the given waypoints
        waypoints: list of waypoints, waypoint: [x, y, yaw] or numpy array of shape (n, 3)
        loop: if True, the path and curve index will be reset to the beginning when reaching the end
    """

    def __init__(
        self,
        receding,
        step_time,
        ref_speed,
        robot,
        waypoints=None,
        loop=False,
        curve_style="line",
        **kwargs,
    ) -> None:

        self.T = receding
        self.dt = step_time
        self.ref_speed = ref_speed
        self.robot = robot
        self.waypoints = self.trans_to_np_list(waypoints)
        self.loop = loop
        self.curve_style = curve_style
        self.min_radius = kwargs.get("min_radius", self.default_turn_radius())
        self.interval = kwargs.get("interval", self.dt * self.ref_speed)
        self.arrive_threshold = kwargs.get("arrive_threshold", 0.1)
        self.close_threshold = kwargs.get("close_threshold", 0.1)
        self.ind_range = kwargs.get("ind_range", 10)
        self.arrive_index_threshold = kwargs.get("arrive_index_threshold", 1)
        self.arrive_flag = False

        self.cg = curve_generator()
        # initial path and gear
        self.initial_path = None

        

    def generate_nom_ref_state(self, state: np.ndarray, cur_vel_array: np.ndarray, ref_speed: float):
        """
        state: current state of the robot, shape (3, 1)
        cur_vel_array: current velocity array of the robot, shape (2, T)
        """
        state = state[:3]

        ref_state = self.cur_point[0:3].copy()
        ref_index = self.point_index
        pre_state = state.copy()

        state_pre_list = [pre_state]
        state_ref_list = [ref_state]

        assert self.cur_point.shape[0] >= 4
        gear_list = [self.cur_point[-1, 0]] * self.T

        ref_speed_forward = ref_speed * self.dt

        for t in range(self.T):
            pre_state = self.motion_predict_model(
                pre_state, cur_vel_array[:, t : t + 1], self.robot.L, self.dt
            )
            state_pre_list.append(pre_state)

            if ref_speed_forward >= self.interval:
                inc_index = int((ref_speed_forward) / self.interval)
                ref_index = ref_index + inc_index

                if ref_index > len(self.cur_curve) - 1:
                    ref_index = len(self.cur_curve) - 1
                    gear_list[t] = 0

                ref_state = self.cur_curve[ref_index][0:3]

            else:
                ref_state, ref_index = self.find_interaction_point(
                    ref_state, ref_index, ref_speed_forward
                )

                if ref_index > len(self.cur_curve) - 1:
                    gear_list[t] = 0

            diff = ref_state[2, 0] - pre_state[2, 0]
            ref_state[2, 0] = pre_state[2, 0] + WrapToPi(diff)
            state_ref_list.append(ref_state)

        nom_s = np.hstack(state_pre_list)
        nom_u = cur_vel_array
        ref_s = np.hstack(state_ref_list)

        # if max(gear_list[1:]) < 0.001:
        #     gear_array = np.zeros(self.T)
        # else:
        gear_array = np.array(gear_list)

        ref_us = gear_array * ref_speed

        return nom_s, nom_u, ref_s, ref_us

    def set_initial_path(self, path):

        '''
        set the initial path from the given path

        Args:
            path: list of points, each point is a numpy array of shape (4, 1)
        '''

        self.initial_path = path
        self.interval = self.cal_average_interval(path)
        self.split_path_with_gear()
        self.curve_index = 0
        self.point_index = 0


    def cal_average_interval(self, path):

        '''
        calculate the average interval of the given path

        Args:
            path: list of points, each point is a numpy array of shape (4, 1)
        '''

        n = len(path)

        if n < 2:
            return 0
        
        dist_sum = 0.0
        for point1, point2 in zip(path, path[1:]):
            x1, y1 = point1[0:2]
            x2, y2 = point2[0:2]
            dist_sum += math.hypot(x2 - x1, y2 - y1)

        return dist_sum / (n - 1)

    def closest_point(self, state, threshold=0.1, ind_range=10):

        min_dis = inf
        cur_index = self.point_index

        start = max(cur_index, 0)
        end = min(cur_index + ind_range, len(self.cur_curve))

        for index in range(start, end):
            dis = distance(state[0:2], self.cur_curve[index][0:2])

            if dis < min_dis:
                min_dis = dis
                self.point_index = index
                if dis < threshold:
                    break

        return min_dis

    def find_interaction_point(self, ref_state, ref_index, length):

        circle = np.squeeze(ref_state[0:2])

        while True:

            if ref_index > len(self.cur_curve) - 2:
                end_point = self.cur_curve[-1]
                end_point[2] = WrapToPi(end_point[2])

                return end_point[0:3], ref_index

            cur_point = self.cur_curve[ref_index]
            next_point = self.cur_curve[ref_index + 1]
            segment = [np.squeeze(cur_point[0:2]), np.squeeze(next_point[0:2])]
            interaction_point = self.range_cir_seg(circle, length, segment)

            if interaction_point is not None:
                diff = WrapToPi(next_point[2, 0] - cur_point[2, 0])
                theta = WrapToPi(cur_point[2, 0] + diff / 2)
                state_ref = np.append(interaction_point, theta).reshape((3, 1))

                return state_ref, ref_index

            else:
                ref_index += 1

    def range_cir_seg(self, circle, r, segment):

        # find the intersection point between the circle and the segment

        assert (
            circle.shape == (2,)
            and segment[0].shape == (2,)
            and segment[1].shape == (2,)
        )

        sp = segment[0]
        ep = segment[1]

        d = ep - sp

        if np.linalg.norm(d) == 0:
            return None

        f = sp - circle

        a = d @ d
        b = 2 * f @ d
        c = f @ f - r**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None
        else:

            # t1 = (-b - sqrt(discriminant)) / (2 * a)
            t2 = (-b + sqrt(discriminant)) / (2 * a)

            if t2 >= 0 and t2 <= 1:
                int_point = sp + t2 * d
                return int_point

            return None

    def check_arrive(
        self,
        state,
    ):

        self.init_check(state)  # check if the initial path is set
        self.closest_point(
            state, self.close_threshold, self.ind_range
        )  # find the closest point on the path

        if self.check_curve_arrive(state, self.arrive_threshold, self.arrive_index_threshold):
            
            if self.curve_index + 1 >= self.curve_number:
                
                if self.loop:
                    self.curve_index = 0
                    self.point_index = 0

                    print("Info: loop, reset the path")
                    # self.initial_path.reverse()
                    # self.split_path_with_gear()
                    return False
                else:
                    if not self.arrive_flag:
                        print("Info: arrive at the end of the path")
                        self.arrive_flag = True
                    return True
            else:
                self.curve_index += 1
                self.point_index = 0

        return False

    def check_curve_arrive(self, state, arrive_threshold=0.1, arrive_index_threshold=2):

        final_point = self.cur_curve[-1][0:2]
        arrive_distance = np.linalg.norm(state[0:2] - final_point)

        return(
            arrive_distance < arrive_threshold
            and self.point_index >= (len(self.cur_curve) - arrive_index_threshold - 2)
        )

    def split_path_with_gear(self):
        """
        split initial path into multiple curves by gear
        """

        if not hasattr(self, "initial_path"):
            raise AttributeError("Object must have a 'initial_path' attribute")

        self.curve_list = []
        current_curve = []
        current_gear = self.initial_path[0][-1]

        for point in self.initial_path:
            if point[-1] != current_gear:
                self.curve_list.append(current_curve)
                current_curve = []
                current_gear = point[-1]

            current_curve.append(point)

        # Append the last curve
        if current_curve:
            self.curve_list.append(current_curve)

    def init_path_with_state(self, state):

        assert len(self.waypoints) > 0, "Error: waypoints are not set"

        if isinstance(self.waypoints, list):
            self.waypoints = [state] + self.waypoints
        elif isinstance(self.waypoints, np.ndarray):
            self.waypoints = np.vstack([state, self.waypoints])

        if self.loop:
            self.waypoints = self.waypoints + [self.waypoints[0]]

        self.initial_path = self.cg.generate_curve(
            self.curve_style, self.waypoints, self.interval, self.min_radius, True
        )

        if self.curve_style == 'line':
            # Ensure consistent angles for line curve
            self._ensure_consistent_angles()

    def init_check(self, state):

        if self.initial_path is None:
            print("initial path is not set, generate path with the current state")
            self.set_ipath_with_state(state)

    def set_ipath_with_state(self, state):

        self.init_path_with_state(state[0:3])
        self.split_path_with_gear()
        # self.path_index = 0
        self.curve_index = 0
        self.point_index = 0

    def update_initial_path_from_goal(self, start, goal):

        if self.loop:
            waypoints = [start, goal, start]
        else:
            waypoints = [start, goal]

        self.initial_path = self.cg.generate_curve(
            self.curve_style, waypoints, self.interval, self.min_radius, True
        )
        
        if self.curve_style == 'line':
            # Ensure consistent angles for line curve
            self._ensure_consistent_angles()

        self.split_path_with_gear()
        # self.path_index = 0
        self.curve_index = 0
        self.point_index = 0
        self.waypoints = waypoints

    def set_ipath_with_waypoints(self, waypoints):
        self.initial_path = self.cg.generate_curve(
            self.curve_style, waypoints, self.interval, self.min_radius, True
        )
        
        if self.curve_style == 'line':
            # Ensure consistent angles for line curve
            self._ensure_consistent_angles()

        self.split_path_with_gear()
        # self.path_index = 0
        self.curve_index = 0
        self.point_index = 0
        self.waypoints = waypoints

    def motion_predict_model(self, robot_state, vel, wheel_base, sample_time):

        if self.robot.kinematics == "acker":
            next_state = self.ackermann_model(robot_state, vel, wheel_base, sample_time)

        elif self.robot.kinematics == "diff":
            next_state = self.diff_model(robot_state, vel, sample_time)
        
        elif self.robot.kinematics == "omni":
            next_state = self.omni_model(robot_state, vel, sample_time)

        return next_state

    def ackermann_model(self, car_state, vel, wheel_base, sample_time):

        assert car_state.shape == (3, 1) and vel.shape == (2, 1)

        phi = car_state[2, 0]

        v = vel[0, 0]
        psi = vel[1, 0]

        ds = np.array([[v * cos(phi)], [v * sin(phi)], [v * tan(psi) / wheel_base]])

        next_state = car_state + ds * sample_time

        # next_state[2, 0] = wraptopi(next_state[2, 0])

        return next_state

    def diff_model(self, robot_state, vel, sample_time):

        assert robot_state.shape == (3, 1) and vel.shape == (2, 1)

        phi = robot_state[2, 0]
        v = vel[0, 0]
        w = vel[1, 0]

        ds = np.array([[v * cos(phi)], [v * sin(phi)], [w]])

        next_state = robot_state + ds * sample_time

        # next_state[2, 0] = wraptopi(next_state[2, 0])

        return next_state
    
    def omni_model(self, robot_state, vel, sample_time):

        assert robot_state.shape[0] >= 2 and vel.shape == (2, 1)

        vx = vel[0, 0] * cos(vel[1, 0])
        vy = vel[0, 0] * sin(vel[1, 0])
        omni_vel = np.array([[vx], [vy], [0]])

        next_state = robot_state + sample_time * omni_vel
       
        return next_state

    @property
    def cur_waypoints(self):
        return self.waypoints

    @property
    def cur_curve(self):
        return self.curve_list[self.curve_index]

    @property
    def cur_point(self):
        return self.cur_curve[self.point_index]

    @property
    def curve_number(self):
        return len(self.curve_list)

    def default_turn_radius(self):

        if self.robot.kinematics == "acker":
            max_psi = self.robot.max_speed[1]
            default_radius = self.robot.L / tan(max_psi)  # radius =  wheelbase / tan(psi)
        else:
            default_radius = 0.0

        return default_radius

    def _ensure_consistent_angles(self):
        """
        Ensure that all points in the initial path have consistent angles.
        For line curves, angles should represent the direction of travel.
        """
        if self.initial_path is None or len(self.initial_path) < 2:
            return
        
        for i in range(len(self.initial_path) - 1):
            current_point = self.initial_path[i]
            next_point = self.initial_path[i + 1]
            
            dx = next_point[0, 0] - current_point[0, 0]
            dy = next_point[1, 0] - current_point[1, 0]
            
            theta = math.atan2(dy, dx)
            
            current_point[2, 0] = theta
        
        if len(self.initial_path) >= 2:
            self.initial_path[-1][2, 0] = self.initial_path[-2][2, 0]

    def trans_to_np_list(self, point_list):

        if point_list is None:
            return []

        return [np.c_[p] if isinstance(p, list) else p for p in point_list]