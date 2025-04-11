import irsim
from neupan import neupan
import torch
import numpy as np

env = irsim.make(display=True)
neupan_planner = neupan.init_from_yaml('planner_0.1.yaml')


def cal_distance_loss(distance, min_distance, collision_threshold, stuck):

    if min_distance <= collision_threshold:
        distance_loss = 50 - torch.sum(distance)
    elif stuck:
        distance_loss = 50 + torch.sum(distance)
    else:
        distance_loss = torch.tensor(0.0, requires_grad=True)

    return distance_loss

def cal_loss(neupan_planner, stuck):
    
    states = neupan_planner.info['state_tensor']
    vel = neupan_planner.info["vel_tensor"]
    distance = neupan_planner.info["distance_tensor"]
    
    ref_state_tensor = neupan_planner.info['ref_state_tensor']
    ref_speed_tensor = neupan_planner.info['ref_speed_tensor']

    state_loss = torch.nn.MSELoss()(states, ref_state_tensor)
    speed_loss = torch.nn.MSELoss()(vel[0, :], ref_speed_tensor)

    distance_loss = cal_distance_loss(distance, neupan_planner.min_distance, neupan_planner.collision_threshold, stuck)

    return state_loss, speed_loss, distance_loss

def train_one_epoch(max_steps=500, render=True):
    
    # total_loss = torch.tensor(0.0)
    loss_of_each_step = []
    opt.zero_grad()

    # pre_position = env.get_robot_state()[0:2]
    # cur_position = pre_position

    stuck_threshold = 0.01
    stack_number_threshold = 5
    stack_number = 0
    arrive_flag = False

    for i in range(max_steps):

        robot_state = env.get_robot_state()[0:3]
        lidar_scan = env.get_lidar_scan()

        points = neupan_planner.scan_to_point(robot_state, lidar_scan)
        action, info = neupan_planner(robot_state, points)

        if render:
            env.render()
        
        pre_position = env.get_robot_state()[0:2]

        env.step(action)

        cur_position = env.get_robot_state()[0:2]
        diff_distance = np.linalg.norm(cur_position - pre_position)

        if diff_distance < stuck_threshold:
            stack_number += 1
        
        if stack_number > stack_number_threshold:
            stuck = True
        else:
            stuck = False

        if stuck:
            print(f'stuck: {stuck}, diff_distance: {diff_distance}')

        _, _, distance_loss = cal_loss(neupan_planner, stuck)

        loss = 10 * distance_loss

        if info['arrive']:
            arrive_flag = True
            print("arrive at target")
        
        # if i == max_steps - 1:
        #     loss += torch.tensor(10.0, requires_grad=True)

        loss_of_each_step.append(loss.item())
        # print(f'loss: {loss.item()}')

        loss.backward()
        opt.step()

        env.draw_trajectory(neupan_planner.opt_trajectory, "r", refresh=True)
        env.draw_trajectory(neupan_planner.ref_trajectory, "b", refresh=True)
        if i == 0:
            env.draw_trajectory(neupan_planner.initial_path, traj_type="-k")

        if info['arrive'] or info['stop'] or stuck:
            env.reset()
            neupan_planner.reset()
            break

    loss_of_each_episode.append(loss_of_each_step)

    env.reset()
    neupan_planner.reset()

    return sum(loss_of_each_step) / len(loss_of_each_step), arrive_flag


if __name__ == "__main__":

    robot_state = env.get_robot_state()

    epoch_num = 150

    q_s_tune = neupan_planner.adjust_parameters[0]
    p_u_tune = neupan_planner.adjust_parameters[1]
    eta = neupan_planner.adjust_parameters[2]
    d_max = neupan_planner.adjust_parameters[3]
    d_min = neupan_planner.adjust_parameters[4]

    opt = torch.optim.Adam([p_u_tune, eta, d_max], lr=5e-3)

    mse_loss = torch.nn.MSELoss()

    loss_of_each_step = []
    loss_of_each_episode = []

    for epoch in range(epoch_num):

        render = epoch % 1 == 0

        total_loss, arrive = train_one_epoch(max_steps=400, render=True) 

        print(f'Epoch: {epoch:02d} Total Loss: {total_loss:.3f} q_s: {q_s_tune.item():.3f} p_u: {p_u_tune.item():.3f} eta: {eta.item():.3f} d_max: {d_max.item():.3f} d_min: {d_min.item():.3f}') 

        if total_loss < 0.05 and arrive:
            break


