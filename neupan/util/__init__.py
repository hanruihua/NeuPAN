'''
util file contains the utility functions for NeuPAN.

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

import time
from neupan import configuration
import os
import sys
from math import sqrt, pi, cos, sin
import numpy as np
import neupan

def time_it(name="Function"):
    """
    Decorator to measure function execution time with instance attribute check.

    Args:
        name (str): Function name for logging (default "Function").

    Returns:
        function: Wrapped function with timing.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            wrapper.count += 1
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            wrapper.func_count += 1
            if configuration.time_print:
                print(f"{name} execute time {(end - start):.6f} seconds")
            return result

        wrapper.count = 0
        wrapper.func_count = 0
        return wrapper

    return decorator


def file_check(file_name):
    """
    Check whether a file exists and return its absolute path.

    Args:
        file_name (str): Name of the file to check.
        root_path (str, optional): Root path to use if the file is not found.

    Returns:
        str: Absolute path of the file if found.

    Raises:
        FileNotFoundError: If the file is not found.
    """

    root_path = os.path.dirname(os.path.dirname(neupan.__file__))

    if file_name is None:
        return None
    
    if os.path.exists(file_name):
        abs_file_name = file_name
    elif os.path.exists(sys.path[0] + "/" + file_name):
        abs_file_name = sys.path[0] + "/" + file_name
    elif os.path.exists(os.getcwd() + "/" + file_name):
        abs_file_name = os.getcwd() + "/" + file_name
    else:
        if root_path is None:
            raise FileNotFoundError("File not found: " + file_name)
        else:
            root_file_name = root_path + "/" + file_name
            if os.path.exists(root_file_name):
                abs_file_name = root_file_name
            else:
                raise FileNotFoundError("File not found: " + root_file_name)

    return abs_file_name



def WrapToPi(rad: float, positive: bool = False) -> float:
    '''The function `WrapToPi` transforms an angle in radians to the range [-pi, pi].
    
    Args:

        rad (float): Angle in radians.
            The `rad` parameter in the `WrapToPi` function represents an angle in radians that you want to
        transform to the range [-π, π]. The function ensures that the angle is within this range by wrapping
        it around if it exceeds the bounds.

        positive (bool): Whether to return the positive value of the angle. Useful for angles difference.
    
    Returns:
        The function `WrapToPi(rad)` returns the angle `rad` wrapped to the range [-pi, pi].
    
    '''
    while rad > pi:
        rad = rad - 2 * pi
    while rad < -pi:
        rad = rad + 2 * pi

    return rad if not positive else abs(rad)


def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Compute the distance between two points.

    Args:
        point1 (np.array): First point [x, y] (2x1).
        point2 (np.array): Second point [x, y] (2x1).

    Returns:
        float: Distance between points.
    """
    return sqrt((point1[0, 0] - point2[0, 0]) ** 2 + (point1[1, 0] - point2[1, 0]) ** 2)


def get_transform(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get rotation and translation matrices from state.

    Args:
        state (np.array): State [x, y, theta] (3x1) or [x, y] (2x1).

    Returns:
        tuple: Translation vector and rotation matrix.
    """
    if state.shape == (2, 1):
        rot = np.array([[1, 0], [0, 1]])
        trans = state[0:2]
    else:
        rot = np.array(
            [
                [cos(state[2, 0]), -sin(state[2, 0])],
                [sin(state[2, 0]), cos(state[2, 0])],
            ]
        )
        trans = state[0:2]
    return trans, rot



def gen_inequal_from_vertex(vertex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate inequality constraints for a convex polygon.

    Args:
        vertex (np.array): Vertices of the polygon (2xN).

    Returns:
        tuple: G matrix and h vector for the inequality Gx <= h.
    """
    convex_flag, order = is_convex_and_ordered(vertex)

    if not convex_flag:
        print("The polygon constructed by vertex is not convex.")
        return None, None

    if order == "CW":
        first_point = vertex[:, 0:1]
        rest_points = vertex[:, 1:]
        vertex = np.hstack([first_point, rest_points[:, ::-1]]) 

        
    num = vertex.shape[1]

    G = np.zeros((num, 2))
    h = np.zeros((num, 1))

    for i in range(num):
        if i + 1 < num:
            pre_point = vertex[:, i]
            next_point = vertex[:, i + 1]
        else:
            pre_point = vertex[:, i]
            next_point = vertex[:, 0]

        diff = next_point - pre_point

        a = diff[1]
        b = -diff[0]
        c = a * pre_point[0] + b * pre_point[1]

        G[i, 0] = a
        G[i, 1] = b
        h[i, 0] = c

    return G, h


def is_convex_and_ordered(points):
    """
    Determine if the polygon is convex and return the order (CW or CCW).

    Args:
        points (np.ndarray): A 2xN NumPy array representing the vertices of the polygon.

    Returns:
        (bool, str): A tuple where the first element is True if the polygon is convex,
                      and the second element is 'CW' or 'CCW' based on the order.
                      If not convex, returns (False, None).
    """
    n = points.shape[1]  # Number of points
    if n < 3:
        return False, None  # A polygon must have at least 3 points

    # Initialize the direction for the first cross product
    direction = 0

    for i in range(n):
        o = points[:, i]
        a = points[:, (i + 1) % n]
        b = points[:, (i + 2) % n]

        cross = cross_product(o, a, b)

        if cross != 0:  # Only consider non-collinear points
            if direction == 0:
                direction = 1 if cross > 0 else -1
            elif (cross > 0 and direction < 0) or (cross < 0 and direction > 0):
                return False, None  # Not convex

    return True, "CCW" if direction > 0 else "CW"


def cross_product(o, a, b):
    """
    Compute the cross product of vectors OA and OB.

    Args:
        o, a, b (array-like): Points representing vectors.

    Returns:
        float: Cross product value.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def repeat_mk_dirs(path, max_num=100):
    """
    Create a directory, appending numbers if it already exists.

    Args:
        path (str): Path of the directory to create.
        max_num (int): Maximum number of attempts.

    Returns:
        str: Path of the created directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    else:
        if len(os.listdir(path)) == 0:  # empty dir
            return path
        else:
            i = 1
            while i < max_num:
                new_path = path + "_" + str(i)
                i = i + 1
                if not os.path.exists(new_path):
                    break
            os.makedirs(new_path)
            return new_path
        

def downsample_decimation(mat, m):
    """
    Downsamples a dim x n matrix to a dim x m matrix using direct sampling uniformly.
    
    Parameters:
        mat: numpy.ndarray of shape (dim, n)
        m: integer, number of columns in the downsampled matrix (m < n)
    
    Returns:
        numpy.ndarray of shape (dim, m)
    """

    n = mat.shape[1]

    if m >= n:
        return mat
    
    indices = np.linspace(0, n - 1, m).astype(int)
    
    sampled_matrix = mat[:, indices]
    return sampled_matrix

