from typing import List, Dict, TYPE_CHECKING, Optional, Union
from gym import spaces
import math
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from PIL import ImageDraw
import time
import sys

from highway_env import utils
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.controller import MDPVehicle

from skimage.transform import resize
from skimage import color

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class ObservationType(object):
    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class HeatmapObservation(ObservationType):
    HALF = int(255 / 2)

    def __init__(self, env: 'AbstractEnv', vehicle=False, clip= True, **config) -> None:
        super().__init__(env)
        self.config = config
        if vehicle:
            self.observer_vehicle = vehicle
        # reading configs
        self.observation_out_shape = None

        self.perception_distance = None
        if "perception_distance" in config:
            self.perception_distance = config["perception_distance"]
        self.speed_embedding = config["speed_embedding"]  # 'log or 'lin' for linear or logarithmic mapping
        self.see_behind = config["see_behind"]
        self.see_behind_ratio = config[
            "see_behind_ratio"]  # this is behind/total 0.2 means less in behind more in front
        self.cooperative_perception = config["cooperative_perception"]
        self.observation_shape = (
            config["observation_shape"][0], config["observation_shape"][1])  # there is a scale factor of 2
        #  TODO fix rotation
        # if "observation_out_shape" in config:
        #     self.observation_out_shape = (config["observation_out_shape"][0],
        #                                   config["observation_out_shape"][1])
        # else:
        #     self.observation_out_shape =  self.observation_shape
        self.observation_out_shape = (
            config["observation_shape"][1], config["observation_shape"][0])

        self.vehicle_amplification = config["vehicle_amplification"]  # length , width
        self.state_features = config["state_features"]  # ["layout", "agents", "humans", "mission", "ego"]
        self.vehicle_speed_range = [vehicle.MIN_SPEED, vehicle.MAX_SPEED]
        self.features_range = {
            "vx": [-vehicle.MIN_SPEED ,vehicle.MAX_SPEED],
            "vy": [-vehicle.MIN_SPEED/2 ,vehicle.MAX_SPEED/2]
        }
        self.history_stack_size = config["history_stack_size"]
        self.conv3D_stack = config["conv3D_stack"]
        self.ego_attention = config["ego_attention"]
        self.absolute = config["absolute"]
        self.map_range = config["map_range"]
        self.flattened = config["flattened"]
        self.diff = config["diff"]
        self.clip = clip
        self.close_vehicles = None
        self.state_road_layout = None
        self.state_agents_box = None
        self.state_humans_box = None
        self.state_mission_box = None
        self.state_ego_box = None
        self.scaling = self.env.road_layout.scaling
        if self.diff and (self.history_stack_size)>1:
            print("ERROR: Diff is not allow with history_stack_size>1")
            sys.exit()
        if self.flattened == False and self.conv3D_stack ==False and (self.history_stack_size)>1:
            print("ERROR: History_stack_size>1 is not allow for not flatten and not conv3D")
            sys.exit()
        if self.history_stack_size == 1:
            if self.flattened:
                self.shape =  (1,) + self.observation_out_shape
            else:
                self.shape = (len(self.state_features),) \
                             + self.observation_out_shape
        else:
            if self.flattened:
                if self.conv3D_stack:
                    self.shape = (1,self.history_stack_size,) \
                                 + self.observation_out_shape
                else:
                    self.shape = (self.history_stack_size,) \
                                 + self.observation_out_shape
            else:
                self.shape = (len(self.state_features),self.history_stack_size,) \
                             + self.observation_out_shape
        self.state = np.zeros(self.shape)
        self.prev_state = np.zeros(self.shape)
        if self.perception_distance == None:
            self.perception_distance = 2 * self.observation_out_shape[1] / self.scaling

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(shape=self.shape,
                              low=0, high=1,
                              dtype=np.float16)
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        start = time.time()
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.cooperative_perception:
            count = None
            see_behind = self.see_behind
        else:
            # here you can change parameters below and order in close_vehicles_to to play with coop perception
            count = 2
            see_behind = False
            self.perception_distance = 50

        self.close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                              distance=self.perception_distance,
                                                              count=count,
                                                              see_behind=see_behind,
                                                              order="sorted")
        self.close_vehicles.append(self.observer_vehicle)

        if "layout" in self.state_features:
            self._plot_state_road_layout()
        if "agents" in self.state_features:
            self._plot_state_agents()
        if "humans" in self.state_features:
            self._plot_state_humans()
        if "mission" in self.state_features:
            self._plot_state_mission()
        if "ego" in self.state_features:
            self._plot_state_ego()

        # TODO fix resize noise
        # if not self.observation_out_shape == None and not self.observation_out_shape == self.observation_shape:
        #     self._resize_state()


        obs_container = []
        if "layout" in self.state_features:
            obs_container.append(self.state_road_layout)
        if "agents" in self.state_features:
            obs_container.append(self.state_agents_box)
        if "humans" in self.state_features:
            obs_container.append(self.state_humans_box)
        if "mission" in self.state_features:
            obs_container.append(self.state_mission_box)
        if "ego" in self.state_features:
            obs_container.append(self.state_ego_box)
        new_obs = np.stack(obs_container, axis=0)


        if self.flattened:
            new_obs = np.sum(new_obs, axis=0)
            # Image.fromarray(new_obs*255).show()
            # Image.fromarray(np.moveaxis(new_obs*255, 0, 1)).show()
            # rotate the image
            new_obs = np.moveaxis(new_obs, 0,1)
            # Image.fromarray(new_obs * 255).show()
            if self.history_stack_size ==1:
                new_obs = np.reshape(new_obs, self.shape)
        else:
            # rotate the image
            new_obs = np.moveaxis(new_obs, [0, 1, 2], [0, -1, -2])

        if self.history_stack_size == 1:
            self.state = new_obs
        else:
            if self.conv3D_stack:
                self.state = np.roll(self.state, -1, axis=1)
                self.state[:,-1, :, :] = new_obs
            else:
                self.state = np.roll(self.state, -1, axis=0)
                self.state[-1, :, :] = new_obs
        # print("size of self.state with float64 = {:.3f} KB".format(sys.getsizeof(self.state) / 1000))
        # TODO :test wihtout converting to int
        self.state = self.state.astype('float16')
        # print("size of self.state with int8 = {:.3f} KB".format(sys.getsizeof(self.state) / 1000))
        # print("HeatmapObservation.observe() in {:.3f}ms".format((time.time() - start) * 1000))

        ## comment this line, this is for debug purposes only
        # utils.visualize_heatmap_state(self)
        if self.flattened and self.diff:
            self.state = new_obs - self.prev_state

            self.prev_state = copy.deepcopy(new_obs)
            self.prev_state = np.reshape(self.prev_state, self.shape)
            #
            # img = np.reshape(self.state, self.observation_out_shape)
            # Image.fromarray(img*255).show()
            # Image.fromarray(np.moveaxis(img*255, 0, 1)).show()
        return self.state

    def _plot_state_road_layout(self):

        surface = self.env.road_layout
        array = surface.layout_array

        observer_position_px = surface.vec2pix(self.observer_vehicle.position)

        if self.see_behind:
            x_range = (observer_position_px[0] - self.see_behind_ratio * self.observation_shape[0]
                       + self.observation_shape[0],
                       observer_position_px[0] + (1 - self.see_behind_ratio) * self.observation_shape[0]
                       + self.observation_shape[0])
            y_range = (observer_position_px[1] - 0.5 * self.observation_shape[1]
                       + self.observation_shape[1],
                       observer_position_px[1] + 0.5 * self.observation_shape[1]
                       + self.observation_shape[1])
        else:
            x_range = (observer_position_px[0] + self.observation_shape[0],
                       observer_position_px[0] + self.observation_shape[0] + self.observation_shape[0])
            y_range = (observer_position_px[1] - 0.5 * self.observation_shape[1]
                       + self.observation_shape[1],
                       observer_position_px[1] + 0.5 * self.observation_shape[1]
                       + self.observation_shape[1])

        array_gray = array[int(x_range[0]):int(x_range[1]), int(y_range[0]):int(y_range[1])]
        array_b_w = array_gray > 200
        array_b_w = array_b_w.astype('float32')
        self.state_road_layout = array_b_w


    def _plot_state_agents(self):
        self.state_agents_box = np.zeros(self.observation_shape)

        for vehicle in self.close_vehicles:
            if vehicle.is_controlled:
                self.state_agents_box = self._plot_vehicle(self.state_agents_box, vehicle)

    def _plot_state_humans(self):
        self.state_humans_box = np.zeros(self.observation_shape)

        for vehicle in self.close_vehicles:
            if not vehicle.is_controlled:
                self.state_humans_box = self._plot_vehicle(self.state_humans_box, vehicle)

    def _plot_state_mission(self):
        self.state_mission_box = np.zeros(self.observation_shape)

        for vehicle in self.close_vehicles:
            # todo: this -1 id should be read from scenario, hard coded for now
            if vehicle.id == -1:
                self.state_mission_box = self._plot_vehicle(self.state_mission_box, vehicle)

    def _plot_state_ego(self):
        self.state_ego_box = np.zeros(self.observation_shape)
        self.state_ego_box = self._plot_vehicle(self.state_ego_box, self.observer_vehicle,
                                                value=1, attention=True)

    def normalize_obs(self,vehicle):
        for feature, f_range in self.features_range.items():
            if feature in vehicle:
                vehicle[feature] = utils.lmap(vehicle[feature], [f_range[0], f_range[1]], self.map_range)
                if self.clip:
                    vehicle[feature] = np.clip(vehicle[feature], self.map_range[0], self.map_range[1])
        return vehicle

    def _plot_vehicle(self, box, vehicle, value=None, attention=False):
        # todo rotate the rectangle using this heading angle

        box_in = copy.deepcopy(box)
        #TODO first normalize , after relative .to_dict
        origin = self.observer_vehicle if not self.absolute else None
        vehicle_dict = vehicle.to_dict(origin, observe_intentions=False)
        # todo fix everything to have a fix_behind ratio
        if self.see_behind:
            x = int(self.see_behind_ratio * self.observation_shape[0] + vehicle_dict['x'] * self.scaling)
            y = int(0.5 * self.observation_shape[1] + vehicle_dict['y'] * self.scaling)
        else:
            x = int(vehicle_dict['x'] * self.scaling)
            y = int(0.5 * self.observation_shape[1] + vehicle_dict['y'] * self.scaling)

        if value == None:
            if self.speed_embedding == 'lin':
                # TODO if vy is not considered then mission vy is not used.
                value = utils.lmap(vehicle_dict['vx'], [-self.vehicle_speed_range[1]/2,
                                                        self.vehicle_speed_range[1]/2], self.map_range)
            elif self.speed_embedding == 'log':
                # TODO Fix log, vx is relative with origin.
                speed = vehicle_dict['vx']
                value = utils.clipped_logmap(speed, 10, self.vehicle_speed_range[1], 0, 1)

            elif self.speed_embedding == 'none':
                value = 1
                
                

        magnification = 1
        if attention:
            magnification = self.ego_attention

        vehicle_length = int(vehicle.LENGTH * self.scaling * self.vehicle_amplification[0] * magnification)
        vehicle_width = int(vehicle.WIDTH * self.scaling * self.vehicle_amplification[1] * magnification)

        box_out = utils.draw_polygon(box_in, x, y, vehicle_length, vehicle_width, vehicle.heading, value , max=1)

        return box_out

    def _resize_state(self):
        out_size = (self.observation_out_shape[1], self.observation_out_shape[0])

        if "layout" in self.state_features:
            self.state_road_layout = np.asarray(
                Image.fromarray(self.state_road_layout).resize(out_size, resample=PIL.Image.BILINEAR))
        if "agents" in self.state_features:
            self.state_agents_box = np.asarray(Image.fromarray(self.state_agents_box).resize(out_size, resample=PIL.Image.BILINEAR))
        if "humans" in self.state_features:
            self.state_humans_box = np.asarray(Image.fromarray(self.state_humans_box).resize(out_size, resample=PIL.Image.BILINEAR))
        if "mission" in self.state_features:
            self.state_mission_box = np.asarray(
                Image.fromarray(self.state_mission_box).resize(out_size, resample=PIL.Image.BILINEAR))
        if "ego" in self.state_features:
            self.state_ego_box = np.asarray(Image.fromarray(self.state_ego_box).resize(out_size, resample=PIL.Image.BILINEAR))


class MultiAgentObservation(ObservationType):
    def __init__(self,
                 env: 'AbstractEnv',
                 observation_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config, vehicle=vehicle)
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.agents_observation_types])

    def observe(self) -> tuple:
        multiagent_observation = tuple(obs_type.observe() for obs_type in self.agents_observation_types)
        # print ("size of multiagent_observation = {:.3f} KB".format(sys.getsizeof(multiagent_observation)/1000))

        return multiagent_observation


def observation_factory(env: 'AbstractEnv', config: dict, vehicle=False) -> ObservationType:
    if config["type"] == "HeatmapObservation":
        return HeatmapObservation(env, vehicle=vehicle, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
