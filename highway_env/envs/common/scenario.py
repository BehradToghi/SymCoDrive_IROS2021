import numpy as np
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.objects import Obstacle
from highway_env.vehicle.behavior import CustomVehicle, CustomVehicleTurn
from highway_env import utils
import random
import copy

class Scenario:
    def __init__(self, env, scenario_number=0):
        self.env = env
        self.road = None
        self.controlled_vehicles = None

        if scenario_number != 0:
            if scenario_number == 2:
                self.env.config.update(self.default_config_merge())
            if scenario_number == 3:
                self.env.config.update(self.default_config_exit())

        self.road_type = self.env.config['scenario']['road_type']

        random_offset = copy.deepcopy(self.env.config['scenario']['random_offset'])
        delta_before, delta_converging, delta_merge = (0, 0, 0)
        if self.env.config['scenario']['randomize_before']:
            delta_before = np.random.randint(low=random_offset[0], high=random_offset[1])
        if self.env.config['scenario']['randomize_converging']:
            delta_converging = np.random.randint(low=random_offset[0], high=random_offset[1])
        if self.env.config['scenario']['randomize_merge']:
            delta_merge = np.random.randint(low=random_offset[0], high=random_offset[1])

        self.before_merging = self.env.config['scenario']['before_merging'] + delta_before
        self.converging_merging = self.env.config['scenario']['converging_merging'] + delta_converging
        self.during_merging = self.env.config['scenario']['during_merging'] + delta_merge
        self.after_merging = self.env.config['scenario']['after_merging']

        self.before_exit = self.env.config['scenario']['before_exit']
        self.converging_exit = self.env.config['scenario']['converging_exit']
        self.taking_exit = self.env.config['scenario']['taking_exit']
        self.during_exit = self.env.config['scenario']['during_exit']
        self.after_exit = self.env.config['scenario']['after_merging']

        self.randomize_vehicles = self.env.config['scenario']['randomize_vehicles']
        self.random_offset_vehicles = copy.deepcopy(self.env.config['scenario']['random_offset_vehicles'])
        self.randomize_speed = self.env.config['scenario']['randomize_speed']
        self.randomize_speed_offset = copy.deepcopy(self.env.config['scenario']['randomize_speed_offset'])
        self.controlled_vehicles_count = self.env.config['controlled_vehicles']
        self.controlled_vehicle_speed = self.env.config['scenario']['controlled_vehicle_speed']

        self.random_controlled_vehicle = self.env.config['scenario']['random_controlled_vehicle']
        # if self.env.config['scenario']['randomize_vehicles']:
        #     self.cruising_vehicles_count_rightmost_lane = self.env.config['vehicles_in_rightmost_lane'] - 1
        #     self.cruising_vehicles_count_other_lanes = self.env.config['vehicles_in_other_lanes']
        # else:
        #     self.cruising_vehicles_count_rightmost_lane = self.env.config['vehicles_count'] - 1

        self.cruising_vehicles_count = self.env.config['vehicles_count'] - 1
        self.cruising_vehicles_front_count = self.env.config['cruising_vehicles_front_count']
        self.cruising_vehicles_front = self.env.config['cruising_vehicles_front']
        self.cruising_vehicles_front_random_everywhere = self.env.config['cruising_vehicles_front_random_everywhere']
        self.cruising_vehicles_front_initial_position = self.env.config['cruising_vehicles_front_initial_position']
        self.total_number_of_vehicles = self.env.config['scenario']['total_number_of_vehicles']
        self.prob_of_controlled_vehicle = self.env.config['scenario']['prob_of_controlled_vehicle']
        self.controlled_baseline_vehicle = self.env.config['controlled_baseline_vehicle']

        if self.env.config['scenario']['random_lane_count']:
            lane_interval = copy.deepcopy(self.env.config['scenario']['lane_count_interval'])
            self.lanes_count = np.random.randint(low=lane_interval[0], high=lane_interval[1])
        else:
            self.lanes_count = self.env.config['lanes_count']

        self.cruising_vehicle = copy.deepcopy({"vehicles_type": self.env.config['cruising_vehicle']["vehicles_type"],
                                 "speed": self.env.config['cruising_vehicle']["speed"],
                                 "enable_lane_change": self.env.config['cruising_vehicle']['enable_lane_change'],
                                 'length': self.env.config['cruising_vehicle']['length']
                                 })

        self.merging_vehicle = copy.deepcopy({'id': self.env.config['merging_vehicle']['id'],
                                'speed': self.env.config['merging_vehicle']['speed'],
                                'initial_position': self.env.config['merging_vehicle']['initial_position'],
                                'random_offset_merging': self.env.config['merging_vehicle']['random_offset_merging'],
                                'controlled_vehicle': self.env.config['merging_vehicle']['controlled_vehicle'],
                                'vehicles_type': self.env.config['merging_vehicle']["vehicles_type"],
                                'set_route': self.env.config['merging_vehicle']['set_route'],
                                'randomize': self.env.config['merging_vehicle']['randomize'],
                                'randomize_speed_merging': self.env.config['merging_vehicle']['randomize_speed_merging'],
                                'min_speed': self.env.config['merging_vehicle']['min_speed'],
                                'max_speed': self.env.config['merging_vehicle'][ 'max_speed'],
                                })

        self.exit_vehicle = copy.deepcopy({'id': self.env.config['exit_vehicle']['id'],
                             'speed': self.env.config['exit_vehicle']['speed'],
                             'initial_position': self.env.config['exit_vehicle']['initial_position'],
                             'controlled_vehicle': self.env.config['exit_vehicle']['controlled_vehicle'],
                             'vehicles_type': self.env.config['exit_vehicle']["vehicles_type"],
                             'set_route': self.env.config['exit_vehicle']['set_route'],
                             'randomize': self.env.config['exit_vehicle']['randomize']
                             })

        self.baseline_vehicle = copy.deepcopy({"vehicles_type": self.env.config['baseline_vehicle']["vehicles_type"],
                                 "speed": self.env.config['baseline_vehicle']["speed"],
                                 "enable_lane_change": self.env.config['baseline_vehicle']['enable_lane_change'],
                                 })

        self.other_vehicles_type = self.env.config["other_vehicles_type"]
        self.record_history = self.env.config["show_trajectories"]
        self.ego_spacing = self.env.config["ego_spacing"]
        self.initial_lane_id = self.env.config["initial_lane_id"]
        self.vehicles_density = self.env.config["vehicles_density"]

        self._create_road(self.road_type)
        self._create_vehicles(self.road_type)

    def create_random(self,cruising_vehicle_class,  from_options =None, speed: float = None, lane_id = None, spacing: float = 1, initial_possition = None, enable_lane_change = True, vehicle_id = 0 , right_lane = None) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        if speed is None:
            speed = self.road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
        default_spacing = 1.5 * speed
        if from_options is None:
            _from = self.road.np_random.choice(list(self.road.network.graph.keys()))
        else:
            _from = self.road.np_random.choice(from_options)

        _to = self.road.np_random.choice(list(self.road.network.graph[_from].keys()))
        if _from == "a" or _from == "b":
            lanes_count = len(self.road.network.graph[_from][_to]) -1
        else:
            lanes_count = len(self.road.network.graph[_from][_to])

        _id = lane_id if lane_id is not None else self.road.np_random.choice(lanes_count)


        # if right_lane:
        #     _id = min(_id, right_lane)
        lane = self.road.network.get_lane((_from, _to, _id))
        offset = spacing * default_spacing * np.exp(-5 / 30 * len(self.road.network.graph[_from][_to]))
        if initial_possition:
            x0 = initial_possition
        else:
            # x0 = np.max([lane.local_coordinates(v.position)[0] for v in self.road.vehicles]) \
            #     if len(self.road.vehicles) else 3 * offset
            distances = []
            for v in self.road.vehicles:
                test = v.lane_index[2]
                if v.lane_index[2] <= lanes_count - 1:
                    distances.append(lane.local_coordinates(v.position)[0])

            x0 = np.max([distances]) if distances else 3 * offset

        x0 += offset * self.road.np_random.uniform(0.9, 1.1)
        x0 = max(0, x0)
        vehicle = cruising_vehicle_class(self.road,
                                         lane.position(x0, 0),
                                         speed=speed, enable_lane_change=enable_lane_change,
                                         config=self.env.config, v_type='cruising_vehicle', id=vehicle_id)

        # v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return vehicle

    def default_config_merge(self) -> dict:
        """
        :return: a configuration dict
        """
        return {

            'duration': 15,  # 40

            'scenario': {
                'scenario_number': 2,
                'road_type': "road_merge",
                # 1-highway, 2-road_closed , 3-road_merge , 4-road_exit, Road types should match with is vehicle_type 1,2,3

                # for merging road
                'lane_count_interval': [1, 4],  # random number of lane range
                'random_offset': [-5, 5],  # offset values for before, converging, merge  -+
                'before_merging': 100,
                'randomize_before': False,  # random before road size
                # distance before converging, converging is the start of the lane with slope
                'converging_merging': 200,
                'randomize_converging': False,  # random converging road size
                # distance from converging to merge, merge start when the slope lane ends
                'during_merging': 110,  # distance of the merging lane, paralles to highway
                'randomize_merge': False,  # random merge road size
                'random_lane_count': False,  # random number of lane
                'after_merging': 1100,  # distance of the highway after that

                # for exit road
                'before_exit': 100,
                'converging_exit': 50,
                'taking_exit': 80,
                'during_exit': 100,
                'after_exit': 1100,

                'randomize_vehicles': True,  # if true vehicles will be randomize based on random_offset_vehicles values
                'random_offset_vehicles': [-5, 5],
                # 'vehicles_in_rightmost_lane':10,  # will overide_vehicle_count if randomize_vehicles = True
                # 'vehicles_in_other_lanes':10,
                'random_controlled_vehicle': False,
                # will chose controlled_vehicle  based on prob_of_controlled_vehicle, override controlled_vehicle
                'total_number_of_vehicles': 13,
                # will be the total number of vehicles in the scenario, AV or cruising will be chosen based on the prob, overide vehicle_count
                'prob_of_controlled_vehicle': 0.5,
                'mission_type': 'merging',

                # if shuffle_controlled_vehicle , from total_number_of_vehicles with probability prob_of_controlled_vehicle AV willl be chosen

            },
            # 'cruising_vehicle': {
            #     'acc_max': 6,  # """Maximum acceleration."""
            #     'comfort_acc_max': 4,  # """Desired maximum acceleration."""
            #     'comfort_acc_min': -12,  # """Desired maximum deceleration."""
            #     'distance_wanted': 0.51,  # """Desired jam distance to the front vehicle."""
            #     'time_wanted': 0.5,  # """Desired time gap to the front vehicle."""
            #     'delta': 4,  # """Exponent of the velocity term."""
            #     'speed': 25,  # Vehicle speed
            #     'enable_lane_change': False,  # allow lane change
            #
            #     'vehicles_type': "highway_env.vehicle.behavior.CustomVehicle",
            #     # chose different vehicle types from :
            #     # "highway_env.vehicle.behavior.CustomVehicle" ,"highway_env.vehicle.behavior.AggressiveVehicle","highway_env.vehicle.behavior.DefensiveVehicle", "highway_env.vehicle.behavior.LinearVehicle"  "highway_env.vehicle.behavior.IDMVehicle"
            #     # if CustomVehicle is chosen it will load the previous configurations, other vehicles types has their own predefiened configurations.
            #     'length': 5.0,  # Vehicle length [m]
            #     'width': 2.0,  # Vehicle width [m]
            #     'max_speed': 40  # Maximum reachable speed [m/s]
            # },

            'merging_vehicle': {
                'acc_max': 6,  # """Maximum acceleration.""" 6
                'comfort_acc_max': 3,  # """Desired maximum acceleration.""" 3
                'comfort_acc_min': -5,  # """Desired maximum deceleration.""" -5
                'distance_wanted': 0.5,  # """Desired jam distance to the front vehicle.""" 5
                'time_wanted': 0.5,  # """Desired time gap to the front vehicle.""" 1.5
                'delta': 4,  # """Exponent of the velocity term.""" 4
                'speed': 25,
                'initial_position': [78, 0],
                'enable_lane_change': False,
                'controlled_vehicle': False,  # chose if merging vehicle is AV or human
                'vehicles_type': "highway_env.vehicle.behavior.CustomVehicle",
                'set_route': False,  # predefine the route
                # "highway_env.vehicle.behavior.CustomVehicle" ,"highway_env.vehicle.behavior.AggressiveVehicle","highway_env.vehicle.behavior.DefensiveVehicle", "highway_env.vehicle.behavior.LinearVehicle"  "highway_env.vehicle.behavior.IDMVehicle"
                # if CustomVehicle is chosen it will load the previous configurations, other vehicles types has their own predefiened configurations.
                'randomize': True,
                'id': -1,  # id for the merging vehicle

                'length': 5.0,  # Vehicle length [m]
                'width': 2.0,  # Vehicle width [m]
                'max_speed': 40  # Maximum reachable speed [m/s]

            },

            "reward": {
                "coop_reward_type": "multi_agent_tuple",
                "reward_type": "merging_reward",  # merging_reward
                "normalize_reward": True,
                "reward_speed_range": [20, 40],
                "collision_reward": -2,  # -1
                "on_desired_lane_reward": 0.3,
                "high_speed_reward": 0.6,  # 0.4
                "lane_change_reward": -0.2,
                "target_lane": 1,
                "distance_reward": -0.1,
                "distance_merged_vehicle_reward": 0,
                "distance_reward_type": "min",
                "successful_merging_reward": 5,
                "continuous_mission_reward": True,
                "cooperative_flag": True,
                "sympathy_flag": True,
                "cooperative_reward": 0.9,
                # True : after merging will keep receiving the reward, False: just received the reward once

            }

        }

    def default_config_exit(self) -> dict:
        """
        :return: a configuration dict
        """
        return {

            'scenario': {
                'scenario_number': 3,
                'road_type': "road_exit",
                # 1-highway, 2-road_closed , 3-road_merge , 4-road_exit, 5-test Road types should match with is vehicle_type 1,2,3

                # for merging road
                'lane_count_interval': [1, 4],  # random number of lane range
                'random_offset': [-5, 5],  # offset values for before, converging, merge  -+
                'before_merging': 100,
                'randomize_before': False,  # random before road size
                # distance before converging, converging is the start of the lane with slope
                'converging_merging': 200,
                'randomize_converging': False,  # random converging road size
                # distance from converging to merge, merge start when the slope lane ends
                'during_merging': 110,  # distance of the merging lane, paralles to highway
                'randomize_merge': False,  # random merge road size
                'random_lane_count': False,  # random number of lane
                'after_merging': 1100,  # distance of the highway after that

                # for exit road
                'before_exit': 100,
                'converging_exit': 50,
                'taking_exit': 40,
                'during_exit': 100,
                'after_exit': 1100,

                'randomize_vehicles': True,  # if true vehicles will be randomize based on random_offset_vehicles values
                'random_offset_vehicles': [-5, 5],
                'random_controlled_vehicle': False,
                # will chose controlled_vehicle  based on prob_of_controlled_vehicle, override controlled_vehicle
                'total_number_of_vehicles': 13,
                # will be the total number of vehicles in the scenario, AV or cruising will be chosen based on the prob, overide vehicle_count
                'prob_of_controlled_vehicle': 0.5,
                'mission_type': 'exit',

                # if shuffle_controlled_vehicle , from total_number_of_vehicles with probability prob_of_controlled_vehicle AV willl be chosen

            },

            # 'cruising_vehicle': {
            #     'acc_max': 6,  # """Maximum acceleration."""
            #     'comfort_acc_max': 4,  # """Desired maximum acceleration."""
            #     'comfort_acc_min': -12,  # """Desired maximum deceleration."""
            #     'distance_wanted': 0.51,  # """Desired jam distance to the front vehicle."""
            #     'time_wanted': 0.5,  # """Desired time gap to the front vehicle."""
            #     'delta': 4,  # """Exponent of the velocity term."""
            #     'speed': 25,  # Vehicle speed
            #     'enable_lane_change': False,  # allow lane change
            #
            #     'vehicles_type': "highway_env.vehicle.behavior.CustomVehicle",
            #     # chose different vehicle types from :
            #     # "highway_env.vehicle.behavior.CustomVehicle" ,"highway_env.vehicle.behavior.AggressiveVehicle","highway_env.vehicle.behavior.DefensiveVehicle", "highway_env.vehicle.behavior.LinearVehicle"  "highway_env.vehicle.behavior.IDMVehicle"
            #     # if CustomVehicle is chosen it will load the previous configurations, other vehicles types has their own predefiened configurations.
            #
            #     'length': 5.0,  # Vehicle length [m]
            #     'width': 2.0,  # Vehicle width [m]
            #     'max_speed': 40  # Maximum reachable speed [m/s]
            # },

            'exit_vehicle': {
                'acc_max': 6,  # """Maximum acceleration.""" 6
                'comfort_acc_max': 3,  # """Desired maximum acceleration.""" 3
                'comfort_acc_min': -5,  # """Desired maximum deceleration.""" -5
                'distance_wanted': 0.5,  # """Desired jam distance to the front vehicle.""" 5
                'time_wanted': 0.5,  # """Desired time gap to the front vehicle.""" 1.5
                'delta': 4,  # """Exponent of the velocity term.""" 4
                'speed': 25,
                'initial_position': [78, 0],
                'enable_lane_change': True,
                'controlled_vehicle': False,  # chose if merging vehicle is AV or human
                'vehicles_type': "highway_env.vehicle.behavior.CustomVehicle",
                'set_route': True,  # predefine the route
                # "highway_env.vehicle.behavior.CustomVehicle" ,"highway_env.vehicle.behavior.AggressiveVehicle","highway_env.vehicle.behavior.DefensiveVehicle", "highway_env.vehicle.behavior.LinearVehicle"  "highway_env.vehicle.behavior.IDMVehicle"
                # if CustomVehicle is chosen it will load the previous configurations, other vehicles types has their own predefiened configurations.
                'randomize': True,
                'id': -1,  # id for the merging vehicle

                'length': 5.0,  # Vehicle length [m]
                'width': 2.0,  # Vehicle width [m]
                'max_speed': 40  # Maximum reachable speed [m/s]
            },

            "reward": {
                "coop_reward_type": "multi_agent_tuple",
                "reward_type": "exit_reward",  # merging_reward
                "normalize_reward": True,
                "reward_speed_range": [20, 40],
                "collision_reward": -2,  # -1
                "on_desired_lane_reward": 0.3,
                "high_speed_reward": 0.6,  # 0.4
                "lane_change_reward": -0.2,
                "target_lane": 1,
                "distance_reward": -0.1,
                "distance_merged_vehicle_reward": 0,
                "distance_reward_type": "min",
                "successful_merging_reward": 5,
                "continuous_mission_reward": True,
                "cooperative_flag": True,
                "sympathy_flag": True,
                "cooperative_reward": 0.9,
                # True : after merging will keep receiving the reward, False: just received the reward once

            }

        }

    def _create_road(self, road_type) -> None:
        if road_type == "highway":
            self._road_highway()
        elif road_type == "road_merge":
            self._road_merge()

        elif road_type == "road_exit":
            self._road_exit()
        elif road_type == "road_closed":
            # TODO , fix arguments
            self._road_closed(end=self.before_merging + self.converging_merging, after=self.after_merging)
        elif road_type == "test":
            self._road_test()

    def _create_vehicles(self, road_type):
        if road_type == "road_merge":
            if self.random_controlled_vehicle:
                self._vehicles_merge_to_highway_prob()
            else:
                self._vehicles_merge_to_highway()
        elif road_type == "road_exit":
            self._vehicles_exit_highway()
        elif road_type == "road_closed":
            # TODO , fix arguments
            self._vehicles_road_closed(controlled_vehicles=self.controlled_vehicles,
                                       cruising_vehicles_count=self.cruising_vehicles_count)
        elif road_type == "highway":
            self._vehicles_highway()
        elif road_type == "test":
            self._vehicle_road_test()

    def _road_highway(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.lanes_count),
                         np_random=self.env.np_random, record_history=self.record_history)

    def _road_merge(self):
        """Create a road composed of straight adjacent lanes."""

        net = RoadNetwork()

        # Highway lanes
        ends = [self.before_merging, self.converging_merging, self.during_merging,
                self.after_merging]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        for lane in range(self.lanes_count):
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == self.lanes_count - 1 else LineType.NONE]

            net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                                [sum(ends[:2]), StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                                line_types=line_types))
            net.add_lane("b", "c",
                         StraightLane([sum(ends[:2]), StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                      [sum(ends[:3]), StraightLane.DEFAULT_WIDTH * (lane + 1)], line_types=line_types))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                                [sum(ends), StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                                line_types=line_types))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + self.lanes_count * 4], [ends[0], 6.5 + 4 + self.lanes_count * 4],
                           line_types=[c, c],
                           forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.env.np_random, record_history=self.record_history)
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))

        self.road = road

    def _road_exit(self):
        """Create a road composed of straight adjacent lanes."""

        net = RoadNetwork()

        # Highway lanes
        ends = [self.before_exit + self.converging_exit, self.taking_exit,
                self.after_exit]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        for lane in range(self.lanes_count):
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == self.lanes_count - 1 else LineType.NONE]

            net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                                [sum(ends[:1]), StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                                line_types=line_types))
            net.add_lane("b", "c",
                         StraightLane([sum(ends[:1]), StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                      [sum(ends[:2]), StraightLane.DEFAULT_WIDTH * (lane + 1)], line_types=line_types))
            net.add_lane("c", "d", StraightLane([sum(ends[:2]), StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                                [sum(ends), StraightLane.DEFAULT_WIDTH * (lane + 1)],
                                                line_types=line_types))

        # Exit lane
        amplitude = 3.25 / 4
        lbp = StraightLane([self.before_exit + self.converging_exit, 4 + self.lanes_count * 4],
                           [self.before_exit + self.converging_exit + self.taking_exit, 4 + self.lanes_count * 4],
                           line_types=[n, c], forbidden=True)

        # ljk = StraightLane([0, 6.5 + 4 +self.lanes_count*4], [ends[0], 6.5 + 4 + self.lanes_count*4 ], line_types=[c, c],
        #                    forbidden=True)
        lpk = SineLane(lbp.position(self.taking_exit, amplitude),
                       lbp.position(self.taking_exit + self.during_exit, amplitude),
                       -amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)

        lkj = StraightLane(lpk.position(self.during_exit, 0), lpk.position(self.during_exit + self.after_exit, 0),
                           line_types=[c, c], forbidden=True)

        net.add_lane("b", "p", lbp)
        net.add_lane("p", "k", lpk)
        net.add_lane("k", "j", lkj)
        road = Road(network=net, np_random=self.env.np_random, record_history=self.record_history)
        # road.objects.append(Obstacle(road, lbp.position(ends[2], 0)))

        self.road = road

    def _road_closed(self, end=200, after=1000):
        """Create a road composed of straight adjacent lanes."""

        net = RoadNetwork()
        last_lane = 0
        # Highway lanes

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [last_lane + StraightLane.DEFAULT_WIDTH, last_lane + 2 * StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]

        new_lane = StraightLane([0, last_lane], [end, last_lane], line_types=[c, n], forbidden=True)
        net.add_lane("a", "b", new_lane)

        for i in range(self.self.lanes_count):
            net.add_lane("a", "b", StraightLane([0, y[i]], [end, y[i]], line_types=line_type[i]))
            net.add_lane("b", "c",
                         StraightLane([end, y[i]], [after, y[i]], line_types=line_type_merge[i]))

        road = Road(network=net, np_random=self.env.np_random, record_history=self.record_history)

        pos = new_lane.position(end, 0)
        road.objects.append(Obstacle(road, pos))
        self.road = road

    def _vehicles_highway(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for _ in range(self.controlled_vehicles_count):
            vehicle = self.env.action_type.vehicle_class.create_random(self.road,
                                                                       speed=25,
                                                                       lane_id=self.initial_lane_id,
                                                                       spacing=self.ego_spacing)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.other_vehicles_type)
        for _ in range(self.cruising_vehicles_count):
            self.road.vehicles.append(
                vehicles_type.create_random(self.road, spacing=1 / self.vehicles_density))

    def _vehicles_road_closed(self, controlled_vehicles=4, cruising_vehicles_count=10) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []

        road = self.road
        vehicle_space = self.converging_merging / controlled_vehicles
        pos = self.before_merging
        for _ in range(controlled_vehicles):
            vehicle = self.env.action_type.vehicle_class(road,
                                                         road.network.get_lane(("a", "b", 1)).position(pos, 0),
                                                         speed=30)
            pos += vehicle_space

            self.controlled_vehicles.append(vehicle)
            road.vehicles.append(vehicle)

        other_vehicles_type = utils.class_from_path(self.other_vehicles_type)

        # spawn vehicles in lane and possition
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(80, 0), speed=29))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=31))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(90, 0), speed=29))

        pos = 0
        vehicle_space = self.before_merging / cruising_vehicles_count
        for i in range(cruising_vehicles_count):
            # spawn vehicles in lane and possition
            road.vehicles.append(
                CustomVehicle(road, road.network.get_lane(("a", "b", 1)).position(pos, 0), config=self.env.config,
                              speed=29,
                              enable_lane_change=False, id=i + 1))
            # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=31))
            pos += vehicle_space

        self.road = road

    def _vehicles_merge_to_highway(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        road = self.road
        right_lane = len(self.road.network.graph['a']['b']) - 1
        vehicle_id = 1

        vehicle_position = 0
        vehicle_space = self.before_merging / self.cruising_vehicles_count
        if vehicle_space <= (abs(self.random_offset_vehicles[0]) + self.cruising_vehicle['length']):
            exit() #comment
            print(" warning, reduce number of vehicle or offset range")
            # TODO , define default for this case
            # TODO , consider speed also for positioning

        cruising_vehicle_class = utils.class_from_path(self.cruising_vehicle["vehicles_type"])
        speed = self.cruising_vehicle["speed"]
        enable_lane_change = self.cruising_vehicle["enable_lane_change"]

        for i in range(self.cruising_vehicles_count):
            if self.randomize_vehicles:
                random_offset = self.random_offset_vehicles
                delta = np.random.randint(low=random_offset[0], high=random_offset[1])

                vehicle_position += delta
                vehicle_position = max(0, vehicle_position)
                # vehicle_position = min(vehicle_position, self.before)
            if self.randomize_speed:
                # speed = road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
                random_offset = self.randomize_speed_offset
                delta = np.random.randint(low=random_offset[0], high=random_offset[1])
                speed += delta
            vehicle = cruising_vehicle_class(road,
                                             road.network.get_lane(("a", "b", right_lane)).position(vehicle_position,
                                                                                                    0),
                                             speed=speed, enable_lane_change=enable_lane_change,
                                             config=self.env.config, v_type='cruising_vehicle', id=vehicle_id)
            vehicle_position += vehicle_space

            road.vehicles.append(vehicle)
            vehicle_id += 1


    # controlled vehicles
        vehicle_space = self.converging_merging / self.controlled_vehicles_count
        vehicle_position = max(vehicle_position + self.cruising_vehicle['length'], self.before_merging + self.random_offset_vehicles[1])

        baseline_vehicle_class = utils.class_from_path(self.baseline_vehicle["vehicles_type"])
        if self.controlled_baseline_vehicle:
            speed = self.baseline_vehicle["speed"]
        else:
            speed = self.controlled_vehicle_speed
        enable_lane_change = self.baseline_vehicle["enable_lane_change"]

        if vehicle_space <= (abs(self.random_offset_vehicles[0]) + self.cruising_vehicle['length']):
            exit()
            print(" warning, reduce number of vehicle or offset range")
            # TODO , define default for this case
            # TODO , consider speed also for positioning

        for _ in range(self.controlled_vehicles_count):
            if self.randomize_vehicles:
                random_offset = self.random_offset_vehicles
                delta = np.random.randint(low=random_offset[0], high=random_offset[1])

                vehicle_position += delta
                vehicle_position = max(0, vehicle_position)

            if self.randomize_speed:
                # speed = road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
                random_offset = self.randomize_speed_offset
                delta = np.random.randint(low=random_offset[0], high=random_offset[1])
                speed += delta

            if self.controlled_baseline_vehicle:
                vehicle = baseline_vehicle_class(road,
                                                 road.network.get_lane(("a", "b", right_lane)).position(
                                                     vehicle_position,
                                                     0),
                                                 speed=speed, enable_lane_change=enable_lane_change,
                                                 config=self.env.config, v_type='baseline_vehicle', id=vehicle_id)
            else:
                vehicle = self.env.action_type.vehicle_class(road,
                                                             road.network.get_lane(("a", "b", right_lane)).position(
                                                                 vehicle_position, 0),
                                                             speed=speed, id=vehicle_id)

            vehicle_position += vehicle_space

            self.controlled_vehicles.append(vehicle)
            road.vehicles.append(vehicle)
            vehicle_id += 1

        if self.cruising_vehicles_front:
            # vehicle_position = max(vehicle_position, self.cruising_vehicles_front_initial_position)
            lane = road.network.get_lane(("b", "c", right_lane))
            last_vehicle_position= lane.local_coordinates(vehicle.position)[0]
            vehicle_position = max(last_vehicle_position + self.ego_spacing * self.cruising_vehicle['length'], self.cruising_vehicles_front_initial_position)
            # vehicle_position = self.cruising_vehicles_front_initial_position
            vehicle_space = self.ego_spacing * self.cruising_vehicle['length']
            enable_lane_change = self.cruising_vehicle["enable_lane_change"]
            speed = self.cruising_vehicle["speed"]
            if vehicle_space <= (abs(self.random_offset_vehicles[0]) + self.cruising_vehicle['length']):
                print(" warning, reduce number of vehicle or offset range")
                exit()
                # TODO , define default for this case
                # TODO , consider speed also for positioning

            for i in range(self.cruising_vehicles_front_count):

                if self.cruising_vehicles_front_random_everywhere:
                    vehicle = self.create_random(cruising_vehicle_class, from_options=["a"],enable_lane_change = self.cruising_vehicle["enable_lane_change"], vehicle_id =vehicle_id)
                else:
                    if self.randomize_vehicles:
                        random_offset = self.random_offset_vehicles
                        delta = np.random.randint(low=random_offset[0], high=random_offset[1])

                        vehicle_position += delta
                        vehicle_position = max(0, vehicle_position)
                        # vehicle_position = min(vehicle_position, self.before)
                    if self.randomize_speed:
                        # speed = road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
                        random_offset = self.randomize_speed_offset
                        delta = np.random.randint(low=random_offset[0], high=random_offset[1])
                        speed += delta

                    vehicle = cruising_vehicle_class(road,
                                                     road.network.get_lane(("b", "c", right_lane)).position(
                                                         vehicle_position,
                                                         0),
                                                     speed=speed, enable_lane_change=self.cruising_vehicle["enable_lane_change"],
                                                     config=self.env.config, v_type='cruising_vehicle', id=vehicle_id)
                    vehicle_position += vehicle_space

                road.vehicles.append(vehicle)
                vehicle_id += 1

        id_merging_vehicle = self.merging_vehicle['id']
        speed = self.merging_vehicle['speed']
        # TODO check everytime we cahnge a var
        initial_position = self.merging_vehicle['initial_position']

        if self.merging_vehicle['randomize']:
            random_offset = self.merging_vehicle['random_offset_merging']
            # delta = np.random.randint(low=random_offset[0], high=random_offset[1])
            delta = np.random.normal(0, random_offset[1] / 3)
            if delta > 0:
                delta = min(delta, random_offset[1])
            else:
                delta = max(delta, -random_offset[1])

            initial_position[0] += delta
            initial_position[0] = max(0, initial_position[0])



        if self.merging_vehicle['randomize_speed_merging']:
            # speed = road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
            random_offset = self.randomize_speed_offset
            # delta = np.random.randint(low=random_offset[0], high=random_offset[1])
            delta = np.random.normal(0, random_offset[1]/3)
            if delta > 0:
                delta = min(delta,random_offset[1])
            else:
                delta = max(delta, -random_offset[1])

            speed += delta
            speed = max(0, speed)

        route = None
        if self.merging_vehicle['controlled_vehicle']:
            # if  self.exit_vehicle['set_route']:
            #     route=[('j', 'k', 0), ('k', 'b', 0), ('b', 'c', 0),('c', 'd', 0)]

            merging_v = self.env.action_type.vehicle_class(road,
                                                           road.network.get_lane(("j", "k", 0)).position(
                                                               initial_position[0],
                                                               initial_position[1]), speed=speed,
                                                           config=self.env.config, id=id_merging_vehicle, route=route , min_speed=self.merging_vehicle['min_speed'], max_speed=self.merging_vehicle['max_speed'])
        else:
            # route = [('j', 'k', 0), ('k', 'b', 0), ('b', 'c', 0), ('c', 'd', 0)]
            merging_vehicle = utils.class_from_path(self.merging_vehicle['vehicles_type'])

            merging_v = merging_vehicle(road, road.network.get_lane(("j", "k", 0)).position(initial_position[0],
                                                                                            initial_position[1]),
                                        speed=speed,
                                        config=self.env.config, v_type='merging_vehicle', id=id_merging_vehicle,
                                        route=route)

        road.vehicles.append(merging_v)
        if self.merging_vehicle['controlled_vehicle']:
            self.controlled_vehicles.append(merging_v)
        self.road = road

    def _vehicles_exit_highway(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        # TODO always in the same possition ??
        # random.seed(30)

        self.controlled_vehicles = []

        road = self.road

        right_lane = len(self.road.network.graph['a']['b']) - 1
        vehicle_space = self.converging_exit / self.controlled_vehicles_count
        vehicle_position = self.before_exit
        vehicle_id = 1
        if self.randomize_vehicles:
            random_offset = self.random_offset_vehicles
            vehicle_position += random_offset[1]

        baseline_vehicle_class = utils.class_from_path(self.baseline_vehicle["vehicles_type"])
        speed = self.baseline_vehicle["speed"]
        enable_lane_change = self.baseline_vehicle["enable_lane_change"]

        for _ in range(self.controlled_vehicles_count):
            if self.controlled_baseline_vehicle:
                vehicle = baseline_vehicle_class(road,
                                                 road.network.get_lane(("a", "b", right_lane)).position(
                                                     vehicle_position,
                                                     0),
                                                 speed=speed, enable_lane_change=enable_lane_change,
                                                 config=self.env.config, v_type='baseline_vehicle', id=vehicle_id)
                vehicle.is_controlled = 1

            else:
                vehicle = self.env.action_type.vehicle_class(road,
                                                             road.network.get_lane(("a", "b", right_lane)).position(
                                                                 vehicle_position, 0),
                                                             speed=30, id=vehicle_id)
            vehicle_position += vehicle_space

            self.controlled_vehicles.append(vehicle)
            road.vehicles.append(vehicle)
            vehicle_id += 1

        vehicle_position = 0
        vehicle_space = self.before_exit / self.cruising_vehicles_count

        cruising_vehicle = utils.class_from_path(self.cruising_vehicle["vehicles_type"])
        # TODO ? speed random?
        speed = self.cruising_vehicle['speed']
        enable_lane_change = self.cruising_vehicle['enable_lane_change']

        for i in range(self.cruising_vehicles_count):
            # spawn vehicles in lane and possition
            # if self.env.config['scenario']['randomize_vehicles']:
            #     # vehicle = cruising_vehicle.create_random(road,spacing=self.env.config["ego_spacing"],id=vehicle_id)
            #     vehicle_position
            # else:
            #     vehicle = cruising_vehicle(road, road.network.get_lane(("a", "b", right_lane)).position(vehicle_position, 0), speed=speed,enable_lane_change=enable_lane_change,
            #                       config=self.env.config,v_type='cruising_vehicle',id=vehicle_id)
            #     vehicle_position += vehicle_space

            if self.randomize_vehicles:
                random_offset = self.random_offset_vehicles
                delta = np.random.randint(low=random_offset[0], high=random_offset[1])

                vehicle_position += delta
                vehicle_position = max(0, vehicle_position)
                # vehicle_position = min(vehicle_position, self.before)
            vehicle = cruising_vehicle(road,
                                       road.network.get_lane(("a", "b", right_lane)).position(vehicle_position, 0),
                                       speed=speed, enable_lane_change=enable_lane_change,
                                       config=self.env.config, v_type='cruising_vehicle', id=vehicle_id)
            vehicle_position += vehicle_space

            road.vehicles.append(vehicle)
            vehicle_id += 1

        id_exit_vehicle = self.exit_vehicle['id']
        speed = self.exit_vehicle['speed']
        initial_position = self.exit_vehicle['initial_position']

        if self.exit_vehicle['randomize']:
            random_offset = self.random_offset_vehicles
            delta = np.random.randint(low=random_offset[0], high=random_offset[1])
            initial_position[0] += delta
            initial_position[0] = max(0, initial_position[0])

        route = None
        if self.exit_vehicle['controlled_vehicle']:
            if self.exit_vehicle['set_route']:
                route = [('a', 'b', 0), ('b', 'p', 0), ('p', 'k', 0), ('k', 'j', 0)]

            exit_v = self.env.action_type.vehicle_class(road,
                                                        road.network.get_lane(("a", "b", 0)).position(
                                                            initial_position[0],
                                                            initial_position[1]), speed=speed, config=self.env.config,
                                                        id=id_exit_vehicle, route=route)
        else:
            exit_vehicle = utils.class_from_path(self.exit_vehicle["vehicles_type"])
            route = [('a', 'b', 0), ('b', 'p', 0), ('p', 'k', 0), ('k', 'j', 0)]
            exit_v = exit_vehicle(road, road.network.get_lane(("a", "b", 0)).position(initial_position[0],
                                                                                      initial_position[1]),
                                  speed=speed,
                                  config=self.env.config, v_type='exit_vehicle', id=id_exit_vehicle,
                                  route=route)

        road.vehicles.append(exit_v)
        if self.merging_vehicle['controlled_vehicle']:
            self.controlled_vehicles.append(exit_v)
        self.road = road

    def _vehicles_merge_to_highway_prob(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []

        road = self.road

        right_lane = len(self.road.network.graph['a']['b']) - 1
        # total_vehicle_count=self.controlled_vehicles_count + self.cruising_vehicles
        total_vehicle_count = self.total_number_of_vehicles
        vehicle_space = (self.converging_merging + self.before_merging) / (total_vehicle_count)
        vehicle_position = 0
        vehicle_id = 1
        prob_of_controlled_vehicle = self.prob_of_controlled_vehicle

        cruising_vehicle = utils.class_from_path(self.cruising_vehicle["vehicles_type"])
        speed = self.cruising_vehicle['speed']
        enable_lane_change = self.cruising_vehicle['enable_lane_change']

        # TODO always in the same possition ??
        random.seed(30)

        for _ in range(total_vehicle_count):
            rand_check = random.random()  # random.uniform(a, b) ,  random.gauss(mu, sigma)
            if self.randomize_vehicles:
                random_offset = self.random_offset_vehicles
                delta = np.random.randint(low=random_offset[0], high=random_offset[1])

                vehicle_position += delta
                vehicle_position = max(0, vehicle_position)

            if rand_check <= prob_of_controlled_vehicle:
                vehicle = self.env.action_type.vehicle_class(road,
                                                             road.network.get_lane(("a", "b", right_lane)).position(
                                                                 vehicle_position, 0),
                                                             speed=30, id=vehicle_id)
                self.controlled_vehicles.append(vehicle)
            else:
                vehicle = cruising_vehicle(road,
                                           road.network.get_lane(("a", "b", right_lane)).position(vehicle_position, 0),
                                           speed=speed, enable_lane_change=enable_lane_change,
                                           config=self.env.config, v_type='cruising_vehicle', id=vehicle_id)
            vehicle_position += vehicle_space

            road.vehicles.append(vehicle)
            vehicle_id += 1

        id_merging_vehicle = self.merging_vehicle['id']
        speed = self.merging_vehicle['speed']
        initial_position = self.merging_vehicle['initial_position']

        if self.merging_vehicle['controlled_vehicle']:
            merging_v = self.env.action_type.vehicle_class(road,
                                                           road.network.get_lane(("j", "k", 0)).position(
                                                               initial_position[0],
                                                               initial_position[1]), speed=speed, id=id_merging_vehicle)
        else:
            merging_vehicle = utils.class_from_path(self.merging_vehicle["vehicles_type"])

            merging_v = merging_vehicle(road, road.network.get_lane(("j", "k", 0)).position(initial_position[0],
                                                                                            initial_position[1]),
                                        speed=speed,
                                        config=self.env.config, v_type='merging_vehicle', id=id_merging_vehicle)

        road.vehicles.append(merging_v)
        if self.merging_vehicle['controlled_vehicle']:
            self.controlled_vehicles.append(merging_v)
        self.road = road

    def _road_test(self) -> None:
        """Test function Rodolfo"""
        # self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
        #                  np_random=self.np_random, record_history=self.config["show_trajectories"])
        net = RoadNetwork()

        lanes = self.lanes_count
        angle = 0
        length = 10000
        for lane in range(lanes):
            origin = np.array([0, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            new_lane = StraightLane(origin, end, line_types=line_types)
            net.add_lane("0", "1", new_lane)

        last_lane = StraightLane.DEFAULT_WIDTH * lanes
        # Highway lanes
        ends = [150, 80, 80, 10000]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [last_lane + StraightLane.DEFAULT_WIDTH, last_lane + 2 * StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]

        new_lane = StraightLane([0, last_lane], [sum(ends[:2]), last_lane], line_types=[c, n], forbidden=True)
        net.add_lane("a", "b", new_lane)

        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c",
                         StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4 + 4 + last_lane], [ends[0], 6.5 + 4 + 4 + 4 + last_lane], line_types=[c, c],
                           forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.env.np_random, record_history=self.record_history)
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))

        pos = new_lane.position(sum(ends[:2]), 0)
        road.objects.append(Obstacle(road, pos))
        self.road = road

    def _vehicle_road_test(self) -> None:
        """Test function Rodolfo"""
        self.controlled_vehicles = []

        road = self.road
        ego_vehicle = self.env.action_type.vehicle_class(road,
                                                         road.network.get_lane(("a", "b", 1)).position(0, 0),
                                                         speed=30)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.other_vehicles_type)

        # spawn vehicles in lane and possition
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=31))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(90, 0), speed=29))

        merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        # self.vehicle = ego_vehicle

        self.controlled_vehicles.append(ego_vehicle)
        self.road = road

        vehicles_type = utils.class_from_path(self.other_vehicles_type)

        # for _ in range(self.config["vehicles_count"]):
        #     self.road.vehicles.append(vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"]))

        # road.vehicles.append(other_vehicles_type(road,road.network.get_lane(("a", "b", 2)).position(100, 0), enable_lane_change=False,speed=31))
        density = 2
        stop_flag = False
        for i in range(6):
            if self.env.config['stopping_vehicle']['stop_flag']:
                stop_flag = True if i == self.env.config['stopping_vehicle']['id'] else False
            self.road.vehicles.append(
                CustomVehicleTurn.create_random_in_a_lane(self.road, ['0', '1', 1], config=self.env.config,
                                                          spacing=1 / density, speed=20, id=i))
        #
        for _ in range(6):
            self.road.vehicles.append(
                CustomVehicle.create_random_in_a_lane(self.road, ['0', '1', 2], config=self.env.config,
                                                      spacing=1 / density, speed=20))
        # for _ in range(self.config["controlled_vehicles_random"]):
        #     vehicle = self.action_type.vehicle_class.create_random(self.road,
        #                                                            speed=25,lane_id=self.config["initial_lane_id"],spacing=self.config["ego_spacing"]) #
        #     self.controlled_vehicles.append(vehicle)
        #     self.road.vehicles.append(vehicle)
