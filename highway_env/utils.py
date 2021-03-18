import copy
import importlib
import itertools
import math
from typing import Tuple, Dict, Callable
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import time
import sys
import numpy as np

from highway_env.types import Vector, Interval
from highway_env.road.road import Road


def do_every(duration: float, timer: float) -> bool:
    return duration < timer


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def clipped_logmap(x, ax, bx, min, max):
    """
    logarithmic mapping of value x to a range from min to max. ax shows the value of x that takes
    the output to 3/4 of the dynamic range and bx is the value that takes the output to the end of
    dynamic range.
    """

    sgn = np.sign(x)
    clipped_x = np.clip(x, 0.35, None)
    y = math.log10(clipped_x / ax * 30) * bx / 80
    y_clipped = sgn * np.clip(y, 0, 1)

    return (y_clipped/2 +0.5) * (max - min) + min


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def constrain(x: float, a: float, b: float) -> np.ndarray:
    return np.clip(x, a, b)


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def point_in_rectangle(point: Vector, rect_min: Vector, rect_max: Vector) -> bool:
    """
    Check if a point is inside a rectangle

    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point: np.ndarray, center: np.ndarray, length: float, width: float, angle: float) \
        -> bool:
    """
    Check if a point is inside a rotated rectangle

    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    :return: is the point inside the rectangle
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, (-length / 2, -width / 2), (length / 2, width / 2))


def point_in_ellipse(point: Vector, center: Vector, angle: float, length: float, width: float) -> bool:
    """
    Check if a point is inside an ellipse

    :param point: a point
    :param center: ellipse center
    :param angle: ellipse main axis angle
    :param length: ellipse big axis
    :param width: ellipse small axis
    :return: is the point inside the ellipse
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(rect1: Tuple[Vector, float, float, float],
                                 rect2: Tuple[Vector, float, float, float]) -> bool:
    """
    Do two rotated rectangles intersect?

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    :return: do they?
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def has_corner_inside(rect1: Tuple[Vector, float, float, float],
                      rect2: Tuple[Vector, float, float, float]) -> bool:
    """
    Check if rect1 has a corner inside rect2

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    (c1, l1, w1, a1) = rect1
    (c2, l2, w2, a2) = rect2
    c1 = np.array(c1)
    l1v = np.array([l1 / 2, 0])
    w1v = np.array([0, w1 / 2])
    r1_points = np.array([[0, 0],
                          - l1v, l1v, -w1v, w1v,
                          - l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1 + np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])


def confidence_ellipsoid(data: Dict[str, np.ndarray], lambda_: float = 1e-5, delta: float = 0.1, sigma: float = 0.1,
                         param_bound: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a confidence ellipsoid over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param lambda_: l2 regularization parameter
    :param delta: confidence level
    :param sigma: noise covariance
    :param param_bound: an upper-bound on the parameter norm
    :return: estimated theta, Gramian matrix G_N_lambda, radius beta_N
    """
    phi = np.array(data["features"])
    y = np.array(data["outputs"])
    g_n_lambda = 1 / sigma * np.transpose(phi) @ phi + lambda_ * np.identity(phi.shape[-1])
    theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.transpose(phi) @ y / sigma
    d = theta_n_lambda.shape[0]
    beta_n = np.sqrt(2 * np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_ ** d) / delta)) + \
             np.sqrt(lambda_ * d) * param_bound
    return theta_n_lambda, g_n_lambda, beta_n


def confidence_polytope(data: dict, parameter_box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute a confidence polytope over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: estimated theta, polytope vertices, Gramian matrix G_N_lambda, radius beta_N
    """
    param_bound = np.amax(np.abs(parameter_box))
    theta_n_lambda, g_n_lambda, beta_n = confidence_ellipsoid(data, param_bound=param_bound)

    values, pp = np.linalg.eig(g_n_lambda)
    radius_matrix = np.sqrt(beta_n) * np.linalg.inv(pp) @ np.diag(np.sqrt(1 / values))
    h = np.array(list(itertools.product([-1, 1], repeat=theta_n_lambda.shape[0])))
    d_theta = np.array([radius_matrix @ h_k for h_k in h])

    # Clip the parameter and confidence region within the prior parameter box.
    theta_n_lambda = np.clip(theta_n_lambda, parameter_box[0], parameter_box[1])
    for k, _ in enumerate(d_theta):
        d_theta[k] = np.clip(d_theta[k], parameter_box[0] - theta_n_lambda, parameter_box[1] - theta_n_lambda)
    return theta_n_lambda, d_theta, g_n_lambda, beta_n


def is_valid_observation(y: np.ndarray, phi: np.ndarray, theta: np.ndarray, gramian: np.ndarray,
                         beta: float, sigma: float = 0.1) -> bool:
    """
    Check if a new observation (phi, y) is valid according to a confidence ellipsoid on theta.

    :param y: observation
    :param phi: feature
    :param theta: estimated parameter
    :param gramian: Gramian matrix
    :param beta: ellipsoid radius
    :param sigma: noise covariance
    :return: validity of the observation
    """
    y_hat = np.tensordot(theta, phi, axes=[0, 0])
    error = np.linalg.norm(y - y_hat)
    eig_phi, _ = np.linalg.eig(phi.transpose() @ phi)
    eig_g, _ = np.linalg.eig(gramian)
    error_bound = np.sqrt(np.amax(eig_phi) / np.amin(eig_g)) * beta + sigma
    return error < error_bound


def is_consistent_dataset(data: dict, parameter_box: np.ndarray = None) -> bool:
    """
    Check whether a dataset {phi_n, y_n} is consistent

    The last observation should be in the confidence ellipsoid obtained by the N-1 first observations.

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: consistency of the dataset
    """
    train_set = copy.deepcopy(data)
    y, phi = train_set["outputs"].pop(-1), train_set["features"].pop(-1)
    y, phi = np.array(y)[..., np.newaxis], np.array(phi)[..., np.newaxis]
    if train_set["outputs"] and train_set["features"]:
        theta, _, gramian, beta = confidence_polytope(train_set, parameter_box=parameter_box)
        return is_valid_observation(y, phi, theta, gramian, beta)
    else:
        return True

def draw_polygon(data, x, y, l, w, h, fill , max=255):
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    rect = get_rect(x=y, y=x, w=w, h=l, angle=h)
    draw.polygon([tuple(p) for p in rect], fill=fill * max)
    # img.show()
    new_data = np.asarray(img)
    return new_data

def get_rect(x, y, w, h, angle):
    # rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    rect = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2)])
    # theta = (np.pi / 180.0) * angle
    theta = angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect

def visualize_heatmap_state(observation , vmax =1):

    # Image.fromarray(np.moveaxis(self.state_road_layout, 0, 1)).show()
    # Image.fromarray(np.moveaxis(self.state_agents_box, 0, 1)).show()
    # Image.fromarray(np.moveaxis(self.state_humans_box, 0, 1)).show()
    # Image.fromarray(np.moveaxis(self.state_mission_box, 0, 1)).show()
    dpi = 96
    stack_size = observation.history_stack_size

    # subplot_start_index = int("6"+str(stack_size)+"1")
    size = [observation.observation_shape[0] * 3 / dpi, observation.observation_shape[1] * 5 * 3 / dpi]
    plt.figure(1, figsize=size)
    plt.suptitle("state for vehicle id #{:n}".format(observation.observer_vehicle.id))
    subplot_start_index = 1

    state = copy.deepcopy(observation.state)
    if stack_size == 1:
        new_shape = (1,) + observation.state.shape
        state = state.reshape(new_shape)

    for layer in state:
        i = subplot_start_index
        j = 0
        if "layout" in observation.state_features:
            state_road_layout = layer[j]
            j += 1
            plt.subplot(6, stack_size, i)
            # plt.imshow(np.moveaxis(state_road_layout, 0, 1), cmap='gray', vmin=0, vmax=vmax)
            plt.imshow(state_road_layout, cmap='gray', vmin=0, vmax=vmax)

            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.title("state_road_layout")
        # subplot_index += stack_size
        i += stack_size
        if "agents" in observation.state_features:
            state_agents_box = layer[j]
            j += 1
            plt.subplot(6, stack_size, i)
            # plt.imshow(np.moveaxis(state_agents_box, 0, 1), cmap='gray', vmin=0, vmax=vmax)
            plt.imshow(state_agents_box, cmap='gray', vmin=0, vmax=vmax)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.title("state_agents_box")
        # subplot_index += stack_size
        i += stack_size
        if "humans" in observation.state_features:
            state_humans_box = layer[j]
            j += 1
            plt.subplot(6, stack_size, i)
            # plt.imshow(np.moveaxis(state_humans_box, 0, 1), cmap='gray', vmin=0, vmax=vmax)
            plt.imshow(state_humans_box, cmap='gray', vmin=0, vmax=vmax)

            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.title("state_humans_box")
        # subplot_index += stack_size
        i += stack_size
        if "mission" in observation.state_features:
            state_mission_box = layer[j]
            j += 1
            plt.subplot(6, stack_size, i)
            # plt.imshow(np.moveaxis(state_mission_box, 0, 1), cmap='gray', vmin=0, vmax=vmax)
            plt.imshow(state_mission_box, cmap='gray', vmin=0, vmax=vmax)

            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.title("state_mission_box")
        # subplot_index += stack_size
        i += stack_size
        if "ego" in observation.state_features:
            state_ego_box = layer[j]
            j += 1
            plt.subplot(6, stack_size, i)
            plt.imshow(state_ego_box, cmap='gray', vmin=0, vmax=vmax)
            # plt.imshow(np.moveaxis(state_ego_box, 0, 1), cmap='gray', vmin=0, vmax=vmax)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.title("attention_map")
        # subplot_index += stack_size
        i += stack_size
        if "layout" in observation.state_features and "humans" in observation.state_features and "agents" in observation.state_features:

            plt.subplot(6, stack_size, i)
            stacked = sum([state_road_layout, state_humans_box, state_agents_box])
            np.clip(stacked, 0, 255)
            plt.imshow(np.moveaxis(stacked, 0, 1), cmap='gray', vmin=0, vmax=vmax)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.title("flattened")
            i += stack_size
        subplot_start_index += 1

    # plt.tight_layout()
    plt.show()

def visualize_image_state(state, figsize = [20,3] , show= True):
    if show is False:
        # Turn interactive plotting off
        plt.ioff()
    plt.grid(False)
    shapes = np.shape(state)
    fig, axs = plt.subplots(shapes[0], 1, figsize=(figsize[0], figsize[1] ))
    for i in range(0, shapes[0]):
        plt.grid(False)
        statei = state [i,:,:]
        # img = Image.fromarray(statei).show()
        if shapes[0]==1:
            axs.imshow(statei, cmap='gray', vmin=0, vmax=1)
        else:
            axs[i].imshow(statei, cmap='gray', vmin=0, vmax=1)
        plt.grid(False)

    # if show:
    #     plt.show()
    #     plt.pause(1 / 15)

def get_road_network_boundaries(road: Road):
    xs = []
    ys = []
    for _from in road.network.graph.keys():
        for _to in road.network.graph[_from].keys():
            for l in road.network.graph[_from][_to]:
                xs.append(l.start[0])
                xs.append(l.end[0])
                ys.append(l.start[1])
                ys.append(l.end[1])

    return [max(xs), max(ys)]