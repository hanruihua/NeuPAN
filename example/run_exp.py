from neupan import neupan
from irsim.env import EnvBase
import numpy as np
import argparse

def main(
    env_file,
    planner_file,
    save_animation=False,
    ani_name="animation",
    full=False,
    no_display=True,
    point_vel=False,
    max_steps=1000, 
):
    
    env = EnvBase(env_file, save_ani=save_animation, full=full, display=no_display)
    neupan_planner = neupan.init_from_yaml(planner_file)
    
    # env.show()

    for i in range(max_steps):

        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()

        if point_vel:
            points, point_velocities = neupan_planner.scan_to_point_velocity(robot_state, lidar_scan)
        else:
            points = neupan_planner.scan_to_point(robot_state, lidar_scan)
            point_velocities = None

        action, info = neupan_planner(robot_state, points, point_velocities)

        if info["stop"]:
            print("NeuPAN stop because of minimum distance")

        if info["arrive"]:
            print("arrive at target")
            break

        env.draw_points(neupan_planner.dune_points, s=25, c="g", refresh=True)
        env.draw_points(neupan_planner.nrmp_points, s=13, c="r", refresh=True)
        env.draw_trajectory(info["opt_state_list"], "r", refresh=True)
        env.draw_trajectory(info["ref_state_list"], "b", refresh=True)

        env.step(action)
        env.render()

        if env.done():
            break

        if i == 0:
            env.draw_trajectory(neupan_planner.initial_path, traj_type="-k")

    env.end(3, ani_name=ani_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--example", type=str, default="polygon_robot", help="pf, pf_obs, corridor, dyna_obs, dyna_non_obs, convex_obs, non_obs, polygon_robot, reverse")
    parser.add_argument("-d", "--kinematics", type=str, default="diff", help="acker, diff")
    parser.add_argument("-a", "--save_animation", action="store_true", help="save animation")
    parser.add_argument("-f", "--full", action="store_true", help="full screen")
    parser.add_argument("-n", "--no_display", action="store_false", help="no display")
    parser.add_argument("-v", "--point_vel", action='store_true', help="point vel")

    args = parser.parse_args()

    env_path_file = args.example + "/" + args.kinematics + "/env.yaml"
    planner_path_file = args.example + "/" + args.kinematics + "/planner.yaml"

    ani_name = args.example + "_" + args.kinematics + "_ani"

    main(env_path_file, planner_path_file, args.save_animation, ani_name, args.full, args.no_display, args.point_vel)
