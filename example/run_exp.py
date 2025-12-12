from neupan import neupan
import irsim
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
    reverse=False,
):
    
    env = irsim.make(env_file, save_ani=save_animation, full=full, display=no_display)
    neupan_planner = neupan.init_from_yaml(planner_file)
    
    # neupan_planner.update_adjust_parameters(q_s=0.5, p_u=1.0, eta=10.0, d_max=1.0, d_min=0.1)
    # neupan_planner.set_reference_speed(5)
    # neupan_planner.update_initial_path_from_waypoints([np.array([0, 0, 0]).reshape(3, 1), np.array([100, 100, 0]).reshape(3, 1)])

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
            print("NeuPAN stops because of minimum distance")

        if info["arrive"]:
            print("NeuPAN arrives at the target")
            break

        env.draw_points(neupan_planner.dune_points, s=25, c="g", refresh=True)
        env.draw_points(neupan_planner.nrmp_points, s=13, c="r", refresh=True)
        env.draw_trajectory(neupan_planner.opt_trajectory, "r", refresh=True)
        env.draw_trajectory(neupan_planner.ref_trajectory, "b", refresh=True)

        env.step(action)
        env.render()

        if env.done():
            break

        if i == 0:
            
            if reverse:
                # for reverse motion
                for j in range(len(neupan_planner.initial_path)):
                    neupan_planner.initial_path[j][-1, 0] = -1
                    neupan_planner.initial_path[j][-2, 0] = neupan_planner.initial_path[j][-2, 0] + 3.14
                
                env.draw_trajectory(neupan_planner.initial_path, traj_type="-k", show_direction=True)
            else:   
                env.draw_trajectory(neupan_planner.initial_path, traj_type="-k", show_direction=False)

            env.render()

    env.end(3, ani_name=ani_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--example", type=str, default="polygon_robot", help="pf, pf_obs, corridor, dyna_obs, dyna_non_obs, convex_obs, non_obs, polygon_robot, reverse")
    parser.add_argument("-d", "--kinematics", type=str, default="diff", help="acker, diff, omni")
    parser.add_argument("-a", "--save_animation", action="store_true", help="save animation")
    parser.add_argument("-f", "--full", action="store_true", help="full screen")
    parser.add_argument("-n", "--no_display", action="store_false", help="no display")
    parser.add_argument("-v", "--point_vel", action='store_true', help="point vel")
    parser.add_argument("-m", "--max_steps", type=int, default=1000, help="max steps")

    args = parser.parse_args()

    env_path_file = args.example + "/" + args.kinematics + "/env.yaml"
    planner_path_file = args.example + "/" + args.kinematics + "/planner.yaml"

    ani_name = args.example + "_" + args.kinematics + "_ani"

    reverse = (args.example == "reverse" and args.kinematics == "diff")

    main(env_path_file, planner_path_file, args.save_animation, ani_name, args.full, args.no_display, args.point_vel, args.max_steps, reverse)
