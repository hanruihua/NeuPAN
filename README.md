<div align="center">

# NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning

<a href="https://arxiv.org/pdf/2403.06828.pdf"><img src='https://img.shields.io/badge/PDF-Arxiv-brightgreen' alt='PDF'></a>
<a href="https://youtu.be/SdSLWUmZZgQ"><img src='https://img.shields.io/badge/Video-Youtube-blue' alt='youtube'></a>
<a href="https://www.bilibili.com/video/BV1Zx421y778/?vd_source=cf6ba629063343717a192a5be9fe8985"><img src='https://img.shields.io/badge/Video-Bilibili-blue' alt='youtube'></a>
<a href="https://hanruihua.github.io/neupan_project/"><img src='https://img.shields.io/badge/Website-NeuPAN-orange' alt='website'></a>

</div>

## News

- **2025-03-25**: Code released!
- **2025-02-04**: Our paper is accepted by **T-RO 2025!**

## Introduction

**NeuPAN** (Neural Proximal Alternating-minimization Network) is an **end-to-end**, **real-time**, **map-free**, and **easy-to-deploy** MPC based robot motion planner. By integrating learning-based and optimization-based techniques, **NeuPAN directly maps obstacle points data to control actions in real-time** by solving an end-to-end mathematical model with numerous point-level collision avoidance constraints. This eliminates middle modules design to avoid error propagation and achieves high accuracy, allowing the robot to navigate in cluttered and unknown environments efficiently and safely.

https://github.com/user-attachments/assets/e37c5775-6e80-4cb5-9320-a04b54792e0e

https://github.com/user-attachments/assets/7e53b88c-aba9-4eea-8708-9bbf0d0305fc





More real world demonstrations are available on the [project page](https://hanruihua.github.io/neupan_project/).

![](./img/Architecture.png)


## Prerequisite
- Python >= 3.10

## Installation

```
git clone https://github.com/hanruihua/NeuPAN
cd NeuPAN
pip install -e .  
```

## Run Examples on IR-SIM

Please Install [IR-SIM](https://github.com/hanruihua/ir-sim) first by:

```
pip install ir-sim
```

You can run examples in the [example](https://github.com/hanruihua/NeuPAN/tree/main/example) folder to see the navigation performance of `diff` (differential) and `acker` (ackermann) robot powered by NeuPAN in IR-SIM. Scenarios include 

`convex_obs` (convex obstacles); `corridor` (corridor); `dyna_non_obs` (dynamic nonconvex obstacles); `dyna_obs` (dynamic obstacles); `non_obs` (nonconvex obstacles); `pf` (path following); `pf_obs` (path following with obstacles); `polygon_robot` (polygon robot), `reverse` (car reverse parking); [dune training](https://github.com/hanruihua/NeuPAN/tree/main/example/dune_train); and [LON training](https://github.com/hanruihua/NeuPAN/tree/main/example/LON). 

Some demonstrations run by `run_exp.py` are shown below:

|     ```python run_exp.py -e non_obs -d acker```  <img src="https://github.com/user-attachments/assets/db88e4c6-2605-4bd4-92b9-8898da732832" width="350" />      | ```python run_exp.py -e dyna_non_obs -d acker```  <img src="https://github.com/user-attachments/assets/4db0594d-f23f-48f0-bf5b-54805c817ac4" width="350" />  |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **```python run_exp.py -e polygon_robot -d diff```**  <img src="https://github.com/user-attachments/assets/feb91992-4d4c-4392-b78d-3553f554667f" width="350" /> | **```python run_exp.py -e dyna_obs -d diff -v```** <img src="https://github.com/user-attachments/assets/43602c29-a0d3-4d06-82e8-500fca8f4aa2" width="350" /> |  |
|   **```python run_exp.py -e corridor -d acker```**  <img src="https://github.com/user-attachments/assets/fbcf9875-2e33-4e97-ba40-54826c2bc70d" width="350" />   |  **```python run_exp.py -e corridor -d diff```**  <img src="https://github.com/user-attachments/assets/82ccd0c5-9ac9-4fcf-8705-d81753c6b7a8" width="350" />  |


> [!NOTE]
> *Since the optimization solver cvxpy is not supported on GPU, we recommend using the CPU device to run the NeuPAN algorithm. Thus, the hardware platform with more powerful CPU is recommended to achieve higher frequency and better performance. However, you can still use the GPU device to train the DUNE model for acceleration.*

## YAML Parameters

Since there are quite a lot of parameters setting for the Neupan planner, we provide a YAML file to initialize the parameters in NeuPAN, which is listed below:

| Parameter Name        | Type / Default Value | Description                                                                          |
| --------------------- | -------------------- | ------------------------------------------------------------------------------------ |
| `receding`            | `int` / 10           | MPC receding steps.                                                                  |
| `step_time`           | `float` / 0.1        | MPC time step (s).                                                                   |
| `ref_speed`           | `float` / 4.0        | MPC reference speed (m/s).                                                           |
| `device`              | `str` / "cpu"        | The device to run the planner.  `cpu` or `cuda`                                      |
| `time_print`          | `bool` / False       | Whether to print the time cost of forward step (s).                                  |
| `collision_threshold` | `float` / 0.1        | The threshold for collision detection (m).                                           |
| `robot`               | `dict` / dict()      | The parameters for the robot. See 'robot' section.                                   |
| `ipath`               | `dict` / dict()      | The parameters for the naive initial path. See 'ipath' section.                      |
| `pan`                 | `dict` / dict()      | The parameters for the proximal alternating minimization network. See 'pan' section. |
| `adjust`              | `dict` / dict()      | The parameters for the adjust weights. See 'adjust' section.                         |
| `train`               | `dict` / dict()      | The parameters for the DUNE training. See 'train' section.                           |

`robot` section:

| Parameter Name | Type / Default Value                       | Description                                                                                                    |
| -------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `kinematics`   | `str` / "diff"                             | The kinematics of the robot. "diff" for differential drive, "acker" for Ackermann drive.                       |
| `vertices`     | `list[list[float]]` / None                 | The vertices of the robot in the initial state. `[[x1, y1], [x2, y2], ...]`                                    |
| `max_speed`    | `list[float]` / [inf, inf]                 | The maximum speed of the robot.                                                                                |
| `max_acce`     | `list[float]` / [inf, inf]                 | The maximum acceleration of the robot.                                                                         |
| `wheelbase`    | `float` / None                             | The wheelbase of the robot. Generally set for the ackermann robots.                                            |
| `length`       | `float` / None                             | The length of the robot. If the `vertices` is not given, this parameter is required for rectangle robot simply |
| `width`        | `float` / None                             | The width of the robot.  If the `vertices` is not given, this parameter is required for rectangle robot simply |
| `name`         | `str` / kinematics + "_robot" + '_default' | The name of the robot. Used for saving the DUNEmodel.                                                          |

`ipath` section:

| Parameter Name           | Type / Default Value                       | Description                                                                                                  |
| ------------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `waypoints`              | `list[list[float]]` / None                 | The waypoints of the path. `[[x1, y1], [x2, y2], ...]`                                                       |
| `loop`                   | `bool` / False                             | When robots arrive at the last waypoint, whether the path will be reset to the beginning.                    |
| `curve_style`            | `str` / "dubin"                            | The style of the curve. "dubin" for Dubin's path, "reedss" for Reeds-Shepp's path. "line" for straight line. |
| `min_radius`             | `float` / default_turn_radius of the robot | The minimum radius of the curve.                                                                             |
| `interval`               | `float` / ref_speed * step_time            | The interval of the points in the path.                                                                      |
| `arrive_threshold`       | `float` / 0.1                              | The threshold to judge whether the robot arrives at the target.                                              |
| `close_threshold`        | `float` / 0.1                              | The threshold to judge the closest point on the path to the robot.                                           |
| `ind_range`              | `int` / 10                                 | The index range of the waypoints, used for finding the next reference point on the path.                     |
| `arrive_index_threshold` | `int` / 1                                  | The threshold of the index to judge whether the robot arrives at the target.                                 |

`pan` section:

| Parameter Name    | Type / Default Value | Description                                                                                          |
| ----------------- | -------------------- | ---------------------------------------------------------------------------------------------------- |
| `iter_num`        | `int` / 2            | The number of iterations. Large number could guarantee convergence but with high computational cost. |
| `dune_max_num`    | `int` / 100          | The maximum number of obstacle points considered in the DUNE layer.                                  |
| `nrmp_max_num`    | `int` / 10           | The maximum number of obstacle points considered in the NRMP layer.                                  |
| `dune_checkpoint` | `str` / None         | The checkpoint model path of the DUNE model.                                                         |
| `iter_threshold`  | `float` / 0.1        | The threshold to judge whether the iteration converges.                                              |

`adjust` section:

*You may adjust the parameters in the `adjust` section to get better performance for your specific workspace.*

| Parameter Name | Type / Default Value | Description                                                                                         |
| -------------- | -------------------- | --------------------------------------------------------------------------------------------------- |
| `q_s`          | `float` / 1.0        | The weight for the state cost. Large value encourages the robot to follow the initial path closely. |
| `p_u`          | `float` / 1.0        | The weight for the speed cost. Large value encourages the robot to follow the reference speed.      |
| `eta`          | `float` / 10.0       | Slack gain for L1 regularization.                                                                   |
| `d_max`        | `float` / 1.0        | The maximum safety distance.                                                                        |
| `d_min`        | `float` / 0.1        | The minimum safety distance.                                                                        |
| `ro_obs`       | `float` / 400        | The penalty parameters for collision avoidance. Smaller value may require more iterations to converge. |
| `bk`           | `float` / 0.1        | The associated proximal coefficient for convergence.                                                 |
| `solver`       | `str` / "ECOS"       | The optimization solver method for the NRMP layer. See [cvxpylayers](https://github.com/cvxgrp/cvxpylayers) and [cvxpy](https://www.cvxpy.org/tutorial/solvers/index.html) for more details. |

## DUNE Model Training for Your Own Robot

To train a DUNE model for your own robot with a specific geometry, you can refer to the [example/dune_train](https://github.com/hanruihua/NeuPAN/tree/main/example/dune_train) folder. Specifically, the geometry is defined in the `robot` section by the `vertices` (or `length` and `width` for rectangle) when the robot is in the initial state (coordinate origin). The training parameters can be adjusted in the `train` section. Generally, the training time is approximately 1-2 hours for a new robot geometry.

> [!NOTE]
> The DUNE model only needs to be trained once for a new robot geometry. This trained model can be used in various environments without retraining.

## ROS Wrapper

We provide a ROS wrapper for NeuPAN. You can refer to the [neupan_ros](https://github.com/hanruihua/neupan_ros) repo to see the details. The Gazebo demonstrations are shown below:

https://github.com/user-attachments/assets/db9edbfe-94d9-4a58-98ee-6b30e64dd3d9

## Citation

If you find our work helpful in your research, you can star our repo and consider citing:

```bibtex

@article{han2024neupan,
  title={NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning},
  author={Han, Ruihua and Wang, Shuai and Wang, Shuaijun and Zhang, Zeqing and Chen, Jianjun and Lin, Shijie and Li, Chengyang and Xu, Chengzhong and Eldar, Yonina C and Hao, Qi and Pan, Jia},
  journal={arXiv preprint arXiv:2403.06828},
  year={2025} 
}
```



