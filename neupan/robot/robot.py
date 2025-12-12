'''
robot class define the robot model and the kinematics model for NeuPAN. It also generate the constraints and cost functions for the optimization problem.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

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
'''

from math import inf
import numpy as np
from typing import Optional, Union
import cvxpy as cp
from math import sin, cos, tan
import torch
from neupan.configuration import to_device 
from neupan.util import gen_inequal_from_vertex

class robot:

    def __init__(
        self,
        receding: int = 10,
        step_time: float = 0.1,
        kinematics: Optional[str] = None,
        vertices: Optional[Union[list[float], np.ndarray]] = None,
        max_speed: list[float] = [inf, inf],
        max_acce: list[float] = [inf, inf],
        wheelbase: Optional[float] = None,
        length: Optional[float] = None,
        width: Optional[float] = None,
        **kwargs,
    ):
        
        if kinematics is None:
            raise ValueError("kinematics is required")

        self.shape = None

        self.vertices = self.cal_vertices(vertices, length, width, wheelbase)
        
        self.G, self.h = gen_inequal_from_vertex(self.vertices)

        self.T = receding
        self.dt = step_time
        self.L = wheelbase

        self.kinematics = kinematics
        self.max_speed = np.c_[max_speed] if isinstance(max_speed, list) else max_speed
        self.max_acce = np.c_[max_acce] if isinstance(max_acce, list) else max_acce

        if kinematics == 'acker':
            if self.max_speed[1] >= 1.57:
                print(f"Warning: max steering angle of acker robot is {self.max_speed[1]} rad, which is larger than 1.57 rad, so it is limited to 1.57 rad")
                self.max_speed[1] = 1.57

        self.speed_bound = self.max_speed
        self.acce_bound = self.max_acce * self.dt

        self.name = kwargs.get("name", self.kinematics + "_robot" + '_default') 

    def define_variable(self, no_obs: bool = False, indep_dis: cp.Variable = None):

        """
        define variables
        """

        self.indep_s = cp.Variable((3, self.T + 1), name="state")  # t0 - T
        self.indep_u = cp.Variable((2, self.T), name="vel")  # t1 - T
        
        indep_list = (
            [self.indep_s, self.indep_u]
            if no_obs
            else [self.indep_s, self.indep_u, indep_dis]
        )

        return indep_list

    def state_parameter_define(self):

        '''
        state parameters:
            - para_gamma_a: q*reference state, 3 * (T+1)
            - para_gamma_b: p*reference speed array, T
            - para_s: nominal state, 3 * (T+1)
            - para_A_list, para_B_list, para_C_list: for state transition model
        '''

        self.para_s = cp.Parameter((3, self.T+1), name='para_state') 
        self.para_gamma_a = cp.Parameter((3, self.T+1), name='para_gamma_a') 
        self.para_gamma_b = cp.Parameter((self.T,), name='para_gamma_b')

        self.para_A_list = [ cp.Parameter((3, 3), name='para_A_'+str(t)) for t in range(self.T)]
        self.para_B_list = [ cp.Parameter((3, 2), name='para_B_'+str(t)) for t in range(self.T)]
        self.para_C_list = [ cp.Parameter((3, 1), name='para_C_'+str(t)) for t in range(self.T)]

        return [self.para_s, self.para_gamma_a, self.para_gamma_b] + self.para_A_list + self.para_B_list + self.para_C_list


    def coefficient_parameter_define(self, no_obs: bool = False, max_num: int = 10):

        """
        gamma_c: lam.T
        zeta_a: lam.T @ p + mu.T @ h
        """

        if no_obs:
            self.para_gamma_c, self.para_zeta_a = [], []

        else:
            self.para_gamma_c = [
                cp.Parameter(
                    (max_num, 2),
                    value=np.zeros((max_num, 2)),
                    name="para_gamma_c" + str(i),
                )
                for i in range(self.T)
            ]  # lam.T, fa
            self.para_zeta_a = [
                cp.Parameter(
                    (max_num, 1),
                    value=np.zeros((max_num, 1)),
                    name="para_zeta_a" + str(i),
                )
                for i in range(self.T)
            ]  # lam.T @ p + mu.T @ h, fb

        return self.para_gamma_c + self.para_zeta_a


    def C0_cost(self, para_p_u, para_q_s):
        
        '''
        reference state cost and control vector cost

        para_p_u: weight of speed cost
        para_q_s: weight of state cost
        '''

        diff_u = para_p_u * self.indep_u[0, :] - self.para_gamma_b
        diff_s = para_q_s * self.indep_s - self.para_gamma_a

        if self.kinematics == 'omni':
            diff_s_cost = cp.sum_squares(diff_s[0:2])
        else:
            diff_s_cost = cp.sum_squares(diff_s)

        C0_cost = diff_s_cost + cp.sum_squares(diff_u) 

        return C0_cost

    def proximal_cost(self):

        """
        proximal cost
        """

        proximal_cost = cp.sum_squares(self.indep_s - self.para_s)

        return proximal_cost


    def I_cost(self, indep_dis, ro_obs):

        cost = 0
        indep_t = self.indep_s[0:2, 1:]

        I_list = []

        for t in range(self.T):

            I_dpp = self.para_gamma_c[t] @ indep_t[:, t:t+1] - self.para_zeta_a[t] - indep_dis[0, t]
            I_list.append(I_dpp)

        I_array = cp.vstack(I_list)
        cost += 0.5 * ro_obs * cp.sum_squares(cp.neg(I_array))

        return cost

    def dynamics_constraint(self):

        '''
        linear dynamics constraints: x_{t+1} = A_t @ x_t + B_t @ u_t + C_t
        '''

        temp_list = []

        for t in range(self.T):
            indep_st = self.indep_s[:, t:t+1]
            indep_ut = self.indep_u[:, t:t+1]

            ## dynamic constraints
            A = self.para_A_list[t]
            B = self.para_B_list[t]
            C = self.para_C_list[t]
            
            temp_list.append(A @ indep_st + B @ indep_ut + C)
        
        constraints = [ self.indep_s[:, 1:] == cp.hstack(temp_list) ]

        return constraints 


    def bound_su_constraints(self):

        '''
        bound constraints on init state, speed, and acceleration   
        '''

        constraints = []

        constraints += [ cp.abs(self.indep_u[:, 1:] - self.indep_u[:, :-1] ) <= self.acce_bound ] 
        constraints += [ cp.abs(self.indep_u) <= self.speed_bound]
        constraints += [ self.indep_s[:, 0:1] == self.para_s[:, 0:1] ]

        return constraints
    

    def generate_state_parameter_value(self, nom_s, nom_u, qs_ref_s, pu_ref_us):

        state_value_list = [nom_s, qs_ref_s, pu_ref_us]

        tensor_A_list = []
        tensor_B_list = []
        tensor_C_list = []

        for t in range(self.T):
            nom_st = nom_s[:, t:t+1]
            nom_ut = nom_u[:, t:t+1]

            if self.kinematics == 'acker':
                A, B, C = self.linear_ackermann_model(nom_st, nom_ut, self.dt, self.L)
            elif self.kinematics == 'diff':
                A, B, C = self.linear_diff_model(nom_st, nom_ut, self.dt)
            elif self.kinematics == 'omni':
                A, B, C = self.linear_omni_model(nom_ut, self.dt)
            else:
                raise ValueError('kinematics currently only supports acker or diff')

            tensor_A_list.append(A)
            tensor_B_list.append(B)
            tensor_C_list.append(C)

        state_value_list += tensor_A_list
        state_value_list += tensor_B_list
        state_value_list += tensor_C_list

        return state_value_list


    
    def linear_ackermann_model(self, nom_st, nom_ut, dt, L):
        
        phi = nom_st[2, 0]
        v, psi = nom_ut[0, 0], nom_ut[1, 0]

        A = torch.Tensor([ [1, 0, -v * dt * sin(phi)], [0, 1, v * dt * cos(phi)], [0, 0, 1] ])

        B = torch.Tensor([ [cos(phi)*dt, 0], [sin(phi)*dt, 0], 
                        [ tan(psi)*dt / L, v*dt/(L * (cos(psi))**2 ) ] ])

        C = torch.Tensor([ [ phi*v*sin(phi)*dt ], [ -phi*v*cos(phi)*dt ], 
                        [ -psi * v*dt / ( L * (cos(psi))**2) ]])
        

        return to_device(A), to_device(B), to_device(C)   
    

    def linear_diff_model(self, nom_state, nom_u, dt):
        
        phi = nom_state[2, 0]
        v = nom_u[0, 0]

        A = torch.Tensor([ [1, 0, -v * dt * sin(phi)], [0, 1, v * dt * cos(phi)], [0, 0, 1] ])

        B = torch.Tensor([ [cos(phi)*dt, 0], [sin(phi)*dt, 0], 
                        [ 0, dt ] ])

        C = torch.Tensor([ [ phi*v*sin(phi)*dt ], [ -phi*v*cos(phi)*dt ], 
                        [ 0 ]])
                
        return to_device(A), to_device(B), to_device(C) 
    
    def linear_omni_model(self, nom_u, dt):
        
        phi = nom_u[1, 0]
        v = nom_u[0, 0]

        A = torch.Tensor([ [1, 0, 0], [0, 1, 0], [0, 0, 1] ])
        B = torch.Tensor([ [ cos(phi) * dt, -v * sin(phi)* dt], [sin(phi)* dt, v*cos(phi) * dt], 
                        [ 0, 0 ] ])

        C = torch.Tensor([ [ phi*v*sin(phi)*dt ], [ -phi*v*cos(phi)*dt ], 
                        [ 0 ]])
        
        return to_device(A), to_device(B), to_device(C) 

    def cal_vertices_from_length_width(self, length, width, wheelbase=None):
        """
        Calculate initial vertices of a rectangle representing a robot.

        Args:
            length (float): Length of the rectangle.
            width (float): Width of the rectangle.
            wheelbase (float): Wheelbase of the robot.

        Returns:
            vertices (np.ndarray): Vertices of the rectangle. shape: (2, 4)
        """
        wheelbase = 0 if wheelbase is None else wheelbase

        start_x = -(length - wheelbase) / 2
        start_y = -width / 2

        point0 = np.array([[start_x], [start_y]])  # left bottom point
        point1 = np.array([[start_x + length], [start_y]])
        point2 = np.array([[start_x + length], [start_y + width]])
        point3 = np.array([[start_x], [start_y + width]])

        return np.hstack((point0, point1, point2, point3))
    
    def cal_vertices(self, vertices = None, length = None, width = None, wheelbase=None):

        '''
        Generate vertices. If vertices is not set, generate vertices from length, width, and wheelbase.

        Args:
            vertices: list of vertices or numpy array of vertices, [[x1, y1], [x2, y2], ...] or (2, N)
            length: length of the robot
            width: width of the robot
            wheelbase: wheelbase of the robot

        Returns:
            vertices_np: numpy array of vertices, shape: (2, N), N >3
        '''

        if vertices is not None:
           if isinstance(vertices, list):
                vertices_np = np.array(vertices).T

           elif isinstance(vertices, np.ndarray):
                vertices_np = vertices
           else:
                raise ValueError("vertices must be a list or numpy array")
           
        else:
            self.shape = "rectangle"
            vertices_np = self.cal_vertices_from_length_width(length, width, wheelbase)
            self.length = length
            self.width = width
            self.wheelbase = wheelbase

        assert vertices_np.shape[1] >= 3, "vertices must be a numpy array of shape (2, N), N >= 3"

        return vertices_np

