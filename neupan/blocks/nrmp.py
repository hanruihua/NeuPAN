"""
NRMP (Neural Regularized Motion Planner) is the core class of the PAN class. It solves the optimization problem integrating the neural latent distance space to generate the optimal control sequence.

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
"""

from typing import Union

import torch
import cvxpy as cp
import numpy as np
from neupan.robot import robot
from neupan.configuration import to_device, value_to_tensor, np_to_tensor, tensor_dtype
from cvxpylayers.torch import CvxpyLayer
from neupan.util import time_it
from typing import Optional, List


class NRMP(torch.nn.Module):

    def __init__(
        self,
        receding: int,
        step_time: float,
        robot: robot,
        nrmp_max_num: int = 10,
        eta: float = 10.0,
        d_max: float = 1.0,
        d_min: float = 0.1,
        q_s: Union[float, list, np.ndarray] = 1.0,
        p_u: float = 1.0,
        ro_obs: float = 400,
        bk: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Initialize the NRMP layer.

        Args:
            receding: int, the number of steps in the receding horizon
            step_time: float, the time step in the MPC framework
            robot: robot, the robot instance including the robot information
            nrmp_max_num: int, the maximum number of obstacle points considered in the NRMP layer
            eta: float, slack gain for L1 regularization
            d_max: float, the maximum safety distance
            d_min: float, the minimum safety distance
            q_s: float or list[float] or np.ndarray, the weight of the state cost.
                 Can be a scalar (e.g., 1.0) or a 3-element list/array for x, y, theta dimensions (e.g., [1.0, 1.0, 0.5])
            p_u: float, the weight of the speed cost
            ro_obs: float, the penalty parameters for collision avoidance
            bk: float, the associated proximal coefficient for convergence
        """
        super(NRMP, self).__init__()

        self.T = receding
        self.dt = step_time
        self.robot = robot
        self.G = np_to_tensor(robot.G)
        self.h = np_to_tensor(robot.h)

        self.max_num = nrmp_max_num
        self.no_obs = False if nrmp_max_num > 0 else True

        # adjust parameters
        self.eta = value_to_tensor(eta, True)
        self.d_max = value_to_tensor(d_max, True)
        self.d_min = value_to_tensor(d_min, True)

        # Convert q_s to tensor, supporting both scalar and 3-element vector
        if isinstance(q_s, (list, np.ndarray)):
            q_s_array = np.array(q_s).flatten()
            if q_s_array.shape[0] != 3:
                raise ValueError(f"q_s must be a scalar or a 3-element list/array, got {q_s_array.shape[0]} elements")
            self.q_s = np_to_tensor(q_s_array, True).reshape(3, 1)            
        else:
            # Scalar case: convert to 3x1 tensor with same value for all dimensions
            self.q_s = value_to_tensor(q_s, True)

        self.p_u = value_to_tensor(p_u, True)

        self.ro_obs = ro_obs
        self.bk = bk

        self.adjust_parameters = (
            [self.q_s, self.p_u]
            if self.no_obs
            else [self.q_s, self.p_u, self.eta, self.d_max, self.d_min]
        )

        # define variables, parameters and problem
        self.variable_definition()
        self.parameter_definition()
        self.problem_definition()

        self.obstacle_points = None
        self.solver = kwargs.get("solver", "ECOS") 

    @time_it("- nrmp forward")
    def forward(
        self,
        nom_s: torch.Tensor,
        nom_u: torch.Tensor,
        ref_s: torch.Tensor,
        ref_us: torch.Tensor,
        mu_list: Optional[List[torch.Tensor]] = None,
        lam_list: Optional[List[torch.Tensor]] = None,
        point_list: Optional[List[torch.Tensor]] = None,
    ):
        """
        nom_s: nominal state, 3 * (T+1)
        nom_u: nominal speed, 1 * T
        ref_s: reference state, 3 * (T+1)
        ref_us: reference speed array, (T,),
        mu_list: list of mu matrix, (max_num, )
        lam_list: list of lam matrix, (max_num, 1)
        point_list: list of obstacle points, (max_num, 2)
        """

        if point_list:
            self.obstacle_points = point_list[0][
                :, : self.max_num
            ]  # current obstacle points considered in the optimization

        parameter_values = self.generate_parameter_value(
            nom_s, nom_u, ref_s, ref_us, mu_list, lam_list, point_list
        )

        solutions = self.nrmp_layer(*parameter_values, solver_args={"solve_method": self.solver}) # see cvxpylayers and cvxpy for more details
        opt_solution_state = solutions[0].type(tensor_dtype)
        opt_solution_vel = solutions[1].type(tensor_dtype)

        nom_d = None if self.no_obs else solutions[2].type(tensor_dtype)

        return opt_solution_state, opt_solution_vel, nom_d

    def generate_parameter_value(
        self, nom_s, nom_u, ref_s, ref_us, mu_list, lam_list, point_list
    ):
        
        adjust_value_list = self.generate_adjust_parameter_value()

        state_value_list = self.robot.generate_state_parameter_value(
            nom_s, nom_u, self.q_s * ref_s, self.p_u * ref_us
        )

        coefficient_value_list = self.generate_coefficient_parameter_value(
            mu_list, lam_list, point_list
        )

        return state_value_list + coefficient_value_list + adjust_value_list

    def generate_adjust_parameter_value(self):
        return self.adjust_parameters

    def update_adjust_parameters_value(self, **kwargs):

        '''
        Update the adjust parameters value: q_s, p_u, eta, d_max, d_min

        Args:
            q_s: float or list[float] or np.ndarray, the weight of the state cost.
                 Can be a scalar or a 3-element list/array for x, y, theta dimensions.
                 Note: The q_s type must be aligned with the initial dimension. If initialized as scalar,
                 only scalar updates are allowed; if initialized as vector, only 3-element vector updates
                 are allowed. Re-initialize the planner to switch types.
            p_u: float, the weight of the speed cost
            eta: float, slack gain for L1 regularization
            d_max: float, the maximum safety distance
            d_min: float, the minimum safety distance
        '''

        q_s_value = kwargs.get("q_s", self.q_s)

        if self.q_s.dim() == 0:
            if isinstance(q_s_value, (list, np.ndarray)):
                value = q_s_value[0]
                print(f"q_s should be a scalar when initialized as scalar, got list/array with {len(q_s_value)} elements. Using the first element: {value}")

            self.q_s = value_to_tensor(value, True)

        elif self.q_s.shape == (3, 1):
            if isinstance(q_s_value, (list, np.ndarray)):
                q_s_array = np.array(q_s_value).flatten()
                if q_s_array.shape[0] != 3:
                    raise ValueError(f"q_s must be a scalar or a 3-element list/array, got {q_s_array.shape[0]} elements")
                np_q_s = q_s_array.reshape(3, 1)
            else:
                raise ValueError(f"q_s must be a 3d list, np.ndarray, or torch.Tensor, got {type(q_s_value)}")

            self.q_s = np_to_tensor(np_q_s, True).reshape(3, 1)

        self.p_u = value_to_tensor(kwargs.get("p_u", self.p_u), True)
        self.eta = value_to_tensor(kwargs.get("eta", self.eta), True)
        self.d_max = value_to_tensor(kwargs.get("d_max", self.d_max), True)
        self.d_min = value_to_tensor(kwargs.get("d_min", self.d_min), True)

        self.adjust_parameters = (
            [self.q_s, self.p_u]
            if self.no_obs
            else [self.q_s, self.p_u, self.eta, self.d_max, self.d_min]
        )


    def generate_coefficient_parameter_value(self, mu_list, lam_list, point_list):
        """
        generate the parameters values for obstacle point avoidance

        Args:
            mu_list: list of mu matrix,
            lam_list: list of lam matrix,
            point_list: list of sorted obstacle points,

        Returns:
            fa_list: list of fa matrix,
            fb_list: list of fb matrix,
        """

        if self.no_obs:
            return []
        else:
            fa_list = [to_device(torch.zeros((self.max_num, 2))) for t in range(self.T)]
            fb_list = [to_device(torch.zeros((self.max_num, 1))) for t in range(self.T)]

            if not mu_list:
                return fa_list + fb_list
            else:
                for t in range(self.T):
                    mu, lam, point = mu_list[t + 1], lam_list[t + 1], point_list[t + 1]
                    fa = lam.T
                    temp = (
                        torch.bmm(lam.T.unsqueeze(1), point.T.unsqueeze(2))
                    ).squeeze(1)

                    fb = temp + mu.T @ self.h

                    # lamb = mu.T @ self.h + torch.matmul(fa.unsqueeze(1), point.T.unsqueeze(2)).squeeze(1)

                    pn = min(mu.shape[1], self.max_num)
                    fa_list[t][:pn, :] = fa[:pn, :]
                    fb_list[t][:pn, :] = fb[:pn, :]

                    fa_list[t][pn:, :] = fa[0, :]
                    fb_list[t][pn:, :] = fb[0, :]

            return fa_list + fb_list

    def variable_definition(self):
        self.indep_dis = cp.Variable(
            (1, self.T), name="distance", nonneg=True
        )  # t1 - T
        self.indep_list = self.robot.define_variable(self.no_obs, self.indep_dis)

    def parameter_definition(self):

        self.para_list = []

        self.para_list += self.robot.state_parameter_define()
        self.para_list += self.robot.coefficient_parameter_define(
            self.no_obs, self.max_num
        )
        self.para_list += self.adjust_parameter_define()

    def problem_definition(self):
        """
        define the optimization problem
        """

        prob = self.construct_prob()

        self.nrmp_layer = to_device(
            CvxpyLayer(prob, parameters=self.para_list, variables=self.indep_list)
        )

    def construct_prob(self):

        nav_cost, nav_constraints = self.nav_cost_cons()
        dune_cost, dune_constraints = self.dune_cost_cons()

        if self.no_obs:
            prob = cp.Problem(cp.Minimize(nav_cost), nav_constraints)
        else:
            prob = cp.Problem(
                cp.Minimize(nav_cost + dune_cost), nav_constraints + dune_constraints
            )

        assert prob.is_dcp(dpp=True)

        return prob

    def adjust_parameter_define(self, **kwargs):
        """
        q and p: the weight of state and control loss, respectively
        eta, d_max, d_min: the parameters for safety distance
        """

        if self.q_s.dim() == 0:
            self.para_q_s = cp.Parameter(name="para_q_s", value=kwargs.get("q_s", 1.0))
        elif self.q_s.shape == (3, 1):
            self.para_q_s = cp.Parameter(
                (3, 1), name="para_q_s", value=kwargs.get("q_s", np.ones((3, 1)))
            )
        self.para_p_u = cp.Parameter(name="para_p_u", value=kwargs.get("p_u", 1.0))

        self.para_eta = cp.Parameter(
            value=kwargs.get("eta", 8), nonneg=True, name="para_eta"
        )
        self.para_d_max = cp.Parameter(
            name="para_d_max", value=kwargs.get("d_max", 1.0), nonneg=True
        )
        self.para_d_min = cp.Parameter(
            name="para_d_min", value=kwargs.get("d_min", 0.1), nonneg=True
        )

        adjust_para_list = (
            [self.para_q_s, self.para_p_u]
            if self.no_obs
            else [
                self.para_q_s,
                self.para_p_u,
                self.para_eta,
                self.para_d_max,
                self.para_d_min,
            ]
        )

        return adjust_para_list

    def nav_cost_cons(self):

        cost = 0
        constraints = []

        cost += self.robot.C0_cost(self.para_p_u, self.para_q_s)
        cost += 0.5 * self.bk * self.robot.proximal_cost()

        constraints += self.robot.dynamics_constraint()
        constraints += self.robot.bound_su_constraints()

        return cost, constraints

    def dune_cost_cons(self):

        cost = 0
        constraints = []

        cost += self.C1_cost_d()  # distance cost

        if not self.no_obs:
            cost += self.robot.I_cost(self.indep_dis, self.ro_obs)

        constraints += self.bound_dis_constraints()

        return cost, constraints

    def bound_dis_constraints(self):

        constraints = []

        constraints += [self.indep_dis >= self.para_d_min]
        constraints += [self.indep_dis <= self.para_d_max]
        # constraints += [cp.max(self.indep_dis) <= self.para_d_max]
        # constraints += [cp.min(self.indep_dis) >= self.para_d_min]

        return constraints

    def C1_cost_d(self):
        return -self.para_eta * cp.sum(self.indep_dis)

    @property
    def points(self):
        """
        point considered in the nrmp layer

        """

        return self.obstacle_points
