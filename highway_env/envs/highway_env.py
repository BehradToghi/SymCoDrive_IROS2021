import numpy as np
from typing import Tuple
import pygame
from PIL import Image
from gym.envs.registration import register
import time

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.envs.common.reward import RewardFactory
from highway_env.road.state_graphics import WorldSurface, RoadGraphics

from highway_env.envs.common.scenario import Scenario
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.objects import Obstacle
from highway_env.vehicle.behavior import CustomVehicle
from highway_env.envs.common.missions import MissionFactory
import copy

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    RIGHT_LANE_REWARD: float = 0.8  # 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 0.8  # 0.4
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    LANE_CHANGE_REWARD: float = 0
    """The reward received at each lane change action."""

    def default_config(self) -> dict:
        # config = super().default_config()
        config = {}
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },

            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": 1,
            "manual_control": False,
            "real_time_rendering": False,

            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "controlled_vehicles_random": 1,

            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self.mission_accomplished = None
        self.mission_vehicle = None
        self.merged_vehicle = False
        self.merged_counter = 0
        self.mission_type = MissionFactory(self)
        scenario = Scenario(self, scenario_number=self.config['scenario']['scenario_number'])
        self.road = scenario.road
        self.road.options = self.options
        self.road.training = self.training
        if self.config["observation"]["observation_config"]["type"] == "HeatmapObservation":
            self.road_layout = self._get_road_layout()
        self.controlled_vehicles = scenario.controlled_vehicles

    def _reward(self, action: int):
        # Cooperative multi-agent reward
        reward_type = RewardFactory(self, action, self.config["reward"])
        calculated_reward = reward_type.cooperative_reward()
        self.reward_info = reward_type.reward_info

        return calculated_reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or th time is out."""

        # TODO clean this up
        # return self.vehicle.crashed or \
        #     self.steps >= self.config["duration"] or \
        #     (self.config["offroad_terminal"] and not self.vehicle.on_road)

        cooperative_flag = self.config["reward"]['cooperative_flag']
        sympathy_flag = self.config["reward"]['sympathy_flag']
        if (cooperative_flag and sympathy_flag)  or self.training == False:
            # testi = 1
            return any(vehicle.crashed for vehicle in self.road.vehicles) \
                   or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
                   or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        else:
            # TODO improve for coop
            # testi=1
            return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
                   or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
                   or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

    def _info(self, steps):
        info = {'timestep': 0,
                'vehicle_ids': [],
                'reward_ids': [],
                'vehicle_is_controlled': [],
                'vehicle_speeds': [],
                'vehicle_distances': [],
                'mission_accomplished': None,
                'collision_count': None,
                'other_vehicle_info_debug': []}

        collisions = []
        # Keeping track of all (controlled and human) vehicles' speeds
        for controlled_vehicle in self.controlled_vehicles:
            info['reward_ids'].append(controlled_vehicle.id)

        for vehicle in self.road.vehicles:
            info['timestep'] = steps
            info['vehicle_ids'].append(vehicle.id)
            info['vehicle_is_controlled'].append(vehicle.is_controlled)
            info['vehicle_speeds'].append(vehicle.speed)
            info['vehicle_distances'].append(vehicle.distance_to_front)
            collisions.append(vehicle.crashed)
            # TODO:
            other_vehicle_info_debug = {
                'positionx': vehicle.position[0],
                'positiony': vehicle.position[1],
                'position': copy.deepcopy(vehicle.position),
                'heading':vehicle.heading,
                'speed':vehicle.speed,
                'vx': vehicle.speed * np.cos(vehicle.heading),
                'vy': vehicle.speed *np.sin(vehicle.heading),
                'lane_index': vehicle.lane_index
            }
            info['other_vehicle_info_debug'].append(other_vehicle_info_debug)
        # in rewward info is eveything also vehicle id
        info['reward_info'] = self.reward_info
        info['mission_accomplished'] = int(self.mission_accomplished)
        info['collision_count'] = sum(collisions)

        return info

    def _get_road_layout(self):
        # road = self.road
        # surface = np.zeros((900, 300))

        pygame.init()
        pygame.display.set_caption("Highway-env")
        obs_configs = self.config["observation"]["observation_config"]
        scaling = obs_configs["scaling"]


        road_boundaries = utils.get_road_network_boundaries(self.road)
        panel_size = (int(road_boundaries[0]*scaling), int(road_boundaries[1]*scaling*1.5))

        observation_shape = (obs_configs["observation_shape"][0], obs_configs["observation_shape"][1])
        # mode 1: road line mode 2: filled rectangles
        sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size), mode=obs_configs["road_layout_mode"])
        sim_surface.scaling = scaling
        sim_surface.centering_position = [0.5, 0.5]

        RoadGraphics.display(self.road, sim_surface)
        data = pygame.surfarray.array3d(sim_surface)
        layout_array_flattened = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
        # Image.fromarray(np.moveaxis((data), 0, 1)).show()
        layout_array_padded = np.pad(layout_array_flattened,
                                     ((observation_shape[0], observation_shape[0]),
                                      (observation_shape[1], observation_shape[1])),
                                     'constant', constant_values=(0, 0))

        # Image.fromarray(np.moveaxis((layout_array_padded), 0, 1)).show()
        sim_surface.layout_array = layout_array_padded

        return sim_surface


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)
