from typing import Tuple, Union

import numpy as np

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.types import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import RoadObject
import time
from timeit import default_timer as timer

class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 id: int =0,
                 config={},
                 v_type='human_vehicle'):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.config=config
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.id=id
        self.exit_offset = self.LENGTH *3 #offset until is allowed to take the exit, smaleer than that distance  exit can not be taked
    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def safe_turn(self,target_lane_index) -> None:

        for v in self.road.vehicles:
            if v is not self \
                    and v.lane_index != target_lane_index:
                d = self.lane_distance_to(v)
                # d_star = self.desired_gap(self, v)
                # if 0 < d < d_star:
                #     return False
                if abs(d)<12:
                    return False

            if v is not self \
                    and v.lane_index == target_lane_index:
                d = np.linalg.norm(v.position - self.position)
                # d_star = self.desired_gap(self, v)
                if 0 < d < 20:
                    return False

        return True

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        # self.save_action_history(action)
        if self.crashed:
            return
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # Lateral: MOBIL
        high_priority_vehicle = False
        if self.config:
            idv = self.config['merging_vehicle']['id']
        else:
            idv = -1
        if self.id == idv:
            high_priority_vehicle = True
        prev_target_lane_index = self.target_lane_index
        if self.route:
            prev_route = self.route[0]
        self.follow_road(high_priority_vehicle)


        # avoid collisions when folow a predifined route
        if self.id==idv and self.route:
            if prev_target_lane_index!= self.target_lane_index and self.safe_turn(self.target_lane_index)==False:
                self.target_lane_index=prev_target_lane_index
                self.route.insert(0,prev_route)

            if len(self.route) >=2:
                getlane = self.road.network.get_lane(self.route[1])
                after_end = getlane.after_end(self.position + self.exit_offset)
                # if passed the next route/node, not way to come back, route=None
                if after_end == True:
                    self.route = None

        if self.enable_lane_change:
            self.change_lane_policy()

        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # Longitudinal: IDM
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.


    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", 0))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def maximum_speed(self, front_vehicle: Vehicle = None) -> Tuple[float, float]:
        """
        Compute the maximum allowed speed to avoid Inevitable Collision States.

        Assume the front vehicle is going to brake at full deceleration and that
        it will be noticed after a given delay, and compute the maximum speed
        which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed speed, and suggested acceleration
        """
        if not front_vehicle:
            return self.target_speed
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.speed
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Speed control
        self.target_speed = min(self.maximum_speed(front_vehicle), self.target_speed)
        acceleration = self.speed_control(self.target_speed)

        return v_max, acceleration

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_speed = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration

class CustomVehicle(IDMVehicle):
    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 4.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -12.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 1.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 id: int = 0,
                 config={},
                 v_type='cruising_vehicle'
                 ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route,
                         enable_lane_change, timer,id,config)
        if len(config):
            acc_max = config[v_type]['acc_max']
            comfort_acc_max = config[v_type]['comfort_acc_max']
            comfort_acc_min = config[v_type]['comfort_acc_min']
            distance_wanted = config[v_type]['distance_wanted']
            time_wanted = config[v_type]['time_wanted']
            delta = config[v_type]['delta']

            length = config[v_type]['length']
            width = config[v_type]['width']
            max_speed = config[v_type]['max_speed']
            # speed_min
            # speed_max


            self.ACC_MAX = acc_max
            self.COMFORT_ACC_MIN = comfort_acc_min
            self.COMFORT_ACC_MAX = comfort_acc_max
            self.DISTANCE_WANTED = distance_wanted + ControlledVehicle.LENGTH
            self.TIME_WANTED = time_wanted
            self.DELTA = delta

            self.MAX_SPEED = max_speed
            self.LENGTH = length
            self.WIDTH = width

    @classmethod
    def create_random_in_a_lane(cls, road: Road, lane_index, speed: float = None, spacing: float = 1,enable_lane_change=False,config={},id: int =0) \
            -> "Vehicle":

        if speed is None:
            speed = road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
        default_spacing = 1.5*speed
        _from = lane_index[0]
        _to = lane_index[1]
        _id = lane_index[2]
        lane = road.network.get_lane((_from, _to, _id))
        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), heading=lane.heading_at(x0), speed=speed, enable_lane_change=enable_lane_change,config=config, id=id)

        return v

    def create_random(cls, road: Road, lane_index, speed: float = None, spacing: float = 1,enable_lane_change=False,config={},id: int =0) \
            -> "Vehicle":

        if speed is None:
            speed = road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
        default_spacing = 1.5*speed
        _from = lane_index[0]
        _to = lane_index[1]
        _id = lane_index[2]
        lane = road.network.get_lane((_from, _to, _id))
        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), heading=lane.heading_at(x0), speed=speed, enable_lane_change=enable_lane_change,config=config, id=id)

        return v


class CustomVehicleTurn(CustomVehicle):
    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 4.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -12.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 1.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 id: int = 0,
                 config={},
                 v_type='human_vehicle',
                 ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route,
                         enable_lane_change, timer,id,config,v_type)
        self.stop_flag = config['stopping_vehicle']['stop_flag']
        self.first_time = True
        self.stop_at = config['stopping_vehicle']['stop_at']
        self.stop_for =config['stopping_vehicle']['stop_for']  # s
        self.t0 = 0
        self.turn =config['stopping_vehicle']['turn']


    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        self.save_action_history(action)
        if self.crashed:
            return
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        # Lateral: MOBIL

        if self.enable_lane_change:
            self.follow_road()
            self.change_lane_policy()

        # s_v, lat_v = lane.local_coordinates(v.position)

        x, y = self.position

        if x> self.stop_at and self.stop_flag and self.id==5:
            if self.first_time:
                self.t0 = timer()
                self.first_time=False
            t1 = timer() - self.t0
            if t1 > self.stop_for:

                if self.turn =='right':
                    target_lane_index = self.get_right_lane()
                    if self.safe_turn(target_lane_index):
                        self.stop_flag=False
                        self.turn_right()
                elif self.turn =='left':
                    target_lane_index = self.get_left_lane()
                    if self.safe_turn(target_lane_index):
                        self.stop_flag = False
                        self.turn_left()
                # self.turn_right()

            self.timer = 0
            self.speed=0
            action['steering']= 0
            action['acceleration']=0

        else:
            action['steering'] = self.steering_control(self.target_lane_index)
            action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

            # Longitudinal: IDM
            action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                       front_vehicle=front_vehicle,
                                                       rear_vehicle=rear_vehicle)
            # action['acceleration'] = self.recover_from_stop(action['acceleration'])
            action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.


    def turn_left(self) -> None:

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # # Does the MOBIL model recommend a lane change?
            # if self.mobil(lane_index):
            #     self.target_lane_index = lane_index

            if lane_index <self.lane_index:
                self.target_lane_index=lane_index

    def turn_right(self) -> None:
        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # # Does the MOBIL model recommend a lane change?
            # if self.mobil(lane_index):
            #     self.target_lane_index = lane_index

            if lane_index > self.lane_index:
                self.target_lane_index = lane_index

    def get_right_lane(self) -> None:
        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # # Does the MOBIL model recommend a lane change?
            # if self.mobil(lane_index):
            #     self.target_lane_index = lane_index

            if lane_index > self.lane_index:
                return lane_index

    def get_left_lane(self) -> None:
        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # # Does the MOBIL model recommend a lane change?
            # if self.mobil(lane_index):
            #     self.target_lane_index = lane_index

            if lane_index < self.lane_index:
                return lane_index

    def safe_turn(self,target_lane_index) -> None:

        for v in self.road.vehicles:
            if v is not self \
                    and v.lane_index != target_lane_index \
                    and v.target_lane_index == target_lane_index:
                d = self.lane_distance_to(v)
                d_star = self.desired_gap(self, v)
                if 0 < d < d_star:
                    return False

            if v is not self \
                    and v.lane_index == target_lane_index:
                d = np.linalg.norm(v.position - self.position)
                # d_star = self.desired_gap(self, v)
                if 0 < d < 20:
                    return False

        return True

class CustomVehicleAggressive(CustomVehicle):
    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 4.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -12.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 1.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 id: int = 0,
                 config={},
                 v_type='cruising_vehicle',
                 ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route,
                         enable_lane_change, timer, id, config, v_type)

        self.first_time = True
        self.merge_at = config['scenario']['before_merging'] + config['scenario']['converging_merging'] + config['scenario']['during_merging'] *3/4

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        # self.save_action_history(action)
        if self.crashed:
            return
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # Lateral: MOBIL
        high_priority_vehicle = False
        if self.config:
            idv = self.config['merging_vehicle']['id']
        else:
            idv = -1
        if self.id == idv:
            high_priority_vehicle = True
        prev_target_lane_index = self.target_lane_index
        if self.route:
            prev_route = self.route[0]
        self.follow_road(high_priority_vehicle)

        # avoid collisions when folow a predifined route
        if self.id == idv and self.route:
            if prev_target_lane_index != self.target_lane_index and self.safe_turn(self.target_lane_index) == False:
                self.target_lane_index = prev_target_lane_index
                self.route.insert(0, prev_route)

            if len(self.route) >= 2:
                getlane = self.road.network.get_lane(self.route[1])
                after_end = getlane.after_end(self.position + self.exit_offset)
                # if passed the next route/node, not way to come back, route=None
                if after_end == True:
                    self.route = None

        x, y = self.position
        _from, _to, _id = self.target_lane_index
        if _from == "b" and _id==1 and x > self.merge_at:
            self.turn_left()


        elif self.enable_lane_change:
            self.change_lane_policy()


        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # Longitudinal: IDM
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)

        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.

    def turn_left(self) -> None:

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # # Does the MOBIL model recommend a lane change?
            # if self.mobil(lane_index):
            #     self.target_lane_index = lane_index

            if lane_index < self.lane_index:
                self.target_lane_index = lane_index

    def get_left_lane(self) -> None:
        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # # Does the MOBIL model recommend a lane change?
            # if self.mobil(lane_index):
            #     self.target_lane_index = lane_index

            if lane_index < self.lane_index:
                return lane_index
class LinearVehicle(IDMVehicle):

    """A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters."""

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    STEERING_PARAMETERS = [ControlledVehicle.KP_HEADING, ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL]

    ACCELERATION_RANGE = np.array([0.5*np.array(ACCELERATION_PARAMETERS), 1.5*np.array(ACCELERATION_PARAMETERS)])
    STEERING_RANGE = np.array([np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
                               np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5])])

    TIME_WANTED = 2.5

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 data: dict = None,
                 config={},
                 v_type='human_vehicle',
                 id=0

                 ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route,
                         enable_lane_change, timer)
        self.data = data if data is not None else {}
        self.collecting_data = True

    def act(self, action: Union[dict, str] = None):
        if self.collecting_data:
            self.collect_data()
        super().act(action)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua*(self.ACCELERATION_RANGE[1] -
                                                                        self.ACCELERATION_RANGE[0])
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub*(self.STEERING_RANGE[1] - self.STEERING_RANGE[0])

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with a Linear Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - reach the speed of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
        - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return float(np.dot(self.ACCELERATION_PARAMETERS,
                            self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle)))

    def acceleration_features(self, ego_vehicle: ControlledVehicle,
                              front_vehicle: Vehicle = None,
                              rear_vehicle: Vehicle = None) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_speed - ego_vehicle.speed
            d_safe = self.DISTANCE_WANTED + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Linear controller with respect to parameters.

        Overrides the non-linear controller ControlledVehicle.steering_control()

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        return float(np.dot(np.array(self.STEERING_PARAMETERS), self.steering_features(target_lane_index)))

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        A collection of features used to follow a lane

        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.PURSUIT_TAU
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array([utils.wrap_to_pi(lane_future_heading - self.heading) *
                             self.LENGTH / utils.not_zero(self.speed),
                             -lane_coords[1] * self.LENGTH / (utils.not_zero(self.speed) ** 2)])
        return features

    def longitudinal_structure(self):
        # Nominal dynamics: integrate speed
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        # Target speed dynamics
        phi0 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ])
        # Front speed control
        phi1 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 1],
            [0, 0, 0, 0]
        ])
        # Front position control
        phi2 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 1, -self.TIME_WANTED, 0],
            [0, 0, 0, 0]
        ])
        # Disable speed control
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if not front_vehicle or self.speed < front_vehicle.speed:
            phi1 *= 0

        # Disable front position control
        if front_vehicle:
            d = self.lane_distance_to(front_vehicle)
            if d != self.DISTANCE_WANTED + self.TIME_WANTED * self.speed:
                phi2 *= 0
        else:
            phi2 *= 0

        phi = np.array([phi0, phi1, phi2])
        return A, phi

    def lateral_structure(self):
        A = np.array([
            [0, 1],
            [0, 0]
        ])
        phi0 = np.array([
            [0, 0],
            [0, -1]
        ])
        phi1 = np.array([
            [0, 0],
            [-1, 0]
        ])
        phi = np.array([phi0, phi1])
        return A, phi

    def collect_data(self):
        """Store features and outputs for parameter regression."""
        self.add_features(self.data, self.target_lane_index)

    def add_features(self, data, lane_index, output_lane=None):

        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        features = self.acceleration_features(self, front_vehicle, rear_vehicle)
        output = np.dot(self.ACCELERATION_PARAMETERS, features)
        if "longitudinal" not in data:
            data["longitudinal"] = {"features": [], "outputs": []}
        data["longitudinal"]["features"].append(features)
        data["longitudinal"]["outputs"].append(output)

        if output_lane is None:
            output_lane = lane_index
        features = self.steering_features(lane_index)
        out_features = self.steering_features(output_lane)
        output = np.dot(self.STEERING_PARAMETERS, out_features)
        if "lateral" not in data:
            data["lateral"] = {"features": [], "outputs": []}
        data["lateral"]["features"].append(features)
        data["lateral"]["outputs"].append(output)


class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               0.5]


class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               2.0]
