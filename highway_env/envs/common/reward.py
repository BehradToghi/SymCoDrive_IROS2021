import numpy as np
from highway_env.vehicle.controller import ControlledVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle
import copy


class RewardFactory():

    def __init__(self, env, action, config):
        self.env = env
        self.action = action
        self.normalize_reward = config["normalize_reward"]
        self.reward_speed_range = copy.deepcopy(config["reward_speed_range"])
        self.coop_reward_type = config["coop_reward_type"]
        self.reward_type = config["reward_type"]
        self.collision_reward = config["collision_reward"]
        self.on_desired_lane_reward = config["on_desired_lane_reward"]
        self.high_speed_reward = config["high_speed_reward"]
        self.lane_change_reward = config["lane_change_reward"]
        self.target_lane = config["target_lane"]
        self.distance_reward = config["distance_reward"]
        self.distance_reward_type = config["distance_reward_type"]
        self.successful_merging_reward = config["successful_merging_reward"]
        self.distance_merged_vehicle_reward = config["distance_merged_vehicle_reward"]
        self.continuous_mission_reward = config['continuous_mission_reward']
        self.continuous_mission_reward_steps_counter = config['continuous_mission_reward_steps_counter']
        self.cooperative_flag = int(config['cooperative_flag'])
        self.sympathy_flag = config['sympathy_flag']
        self.cooperative_reward_value = config['cooperative_reward']
        self.reward_info = []
        self.self_reward = 1
        self.AV_reward = 1
        self.HV_reward = 1

        # if self.sympathy_flag == False and self.env.config["merging_vehicle"]["controlled_vehicle"] == False:
        #     self.successful_merging_reward = 0
        if self.cooperative_flag:
            self.avg_speeds = self.get_avg_speeds(sympathy_flag=self.sympathy_flag, scaled_speed_flag=True)
        else:
            # self.cooperative_reward_value = 0
            # self.successful_merging_reward = 0
            self.avg_speeds = 0
        # if self.target_lane >= self.env.config["lanes_count"]:
        #     raise Exception("The target_lane config cannot be greater than lanes_count-1 config")

    def get_sum_of_speeds(self, sympathy_flag=False, scaled_speed_flag=False):
        sum_of_speeds = 0
        if sympathy_flag:
            for vehicle in self.env.road.vehicles:
                if scaled_speed_flag:
                    scaled_speed = utils.lmap(vehicle.speed, self.reward_speed_range, [0, 1])
                else:
                    scaled_speed = vehicle.speed

                sum_of_speeds += scaled_speed
        else:
            for vehicle in self.env.controlled_vehicles:
                if scaled_speed_flag:
                    scaled_speed = utils.lmap(vehicle.speed, self.reward_speed_range, [0, 1])
                else:
                    scaled_speed = vehicle.speed

                sum_of_speeds += scaled_speed

        return sum_of_speeds

    def get_avg_speeds(self, sympathy_flag=False, scaled_speed_flag=False):
        speeds = []
        if sympathy_flag:
            for vehicle in self.env.road.vehicles:
                if scaled_speed_flag:
                    scaled_speed = utils.lmap(vehicle.speed, self.reward_speed_range, [0, 1])
                else:
                    scaled_speed = vehicle.speed
                speeds.append(scaled_speed)
        else:
            for vehicle in self.env.controlled_vehicles:
                if scaled_speed_flag:
                    scaled_speed = utils.lmap(vehicle.speed, self.reward_speed_range, [0, 1])
                else:
                    scaled_speed_flag = vehicle.speed

                speeds.append(scaled_speed)

        return np.mean(speeds)

    def cooperative_reward(self):
        if self.coop_reward_type == "multi_agent_tuple":
            return self._multi_agent_tuple_reward()
        elif self.coop_reward_type == "linear_sum":
            return self._linear_sum_reward()
        elif self.coop_reward_type == "single_agent":
            return self._single_agent_reward()

    def _single_agent_reward(self):
        if self.reward_type == "type_1":
            # return self._agent_reward_type_1(self.env.controlled_vehicles[0])
            return self._agent_reward_type_1(self.env.vehicle, self.action)
        elif self.reward_type == "type_2":
            return self._agent_reward_type_2(self.env.vehicle, self.action)
        elif self.reward_type == "type_3":
            return self._agent_reward_type_3(self.env.vehicle, self.action)

    def _multi_agent_tuple_reward(self):
        if self.reward_type == "type_1":
            return tuple(self._agent_reward_type_1(_vehicle, _action) for _vehicle, _action in
                         zip(self.env.controlled_vehicles, self.action))
        elif self.reward_type == "type_2":
            return tuple(self._agent_reward_type_2(_vehicle, _action) for _vehicle, _action in
                         zip(self.env.controlled_vehicles, self.action))
        elif self.reward_type == "type_3":
            return tuple(self._agent_reward_type_3(_vehicle, _action) for _vehicle, _action in
                         zip(self.env.controlled_vehicles, self.action))
        elif self.reward_type == "merging_reward":
            reward = []
            reward_info = []
            for _vehicle, _action in zip(self.env.controlled_vehicles, self.action):
                rewardi, reward_infoi = self._agent_reward_merging_reward(_vehicle, _action)
                reward.append(rewardi)
                reward_info.append(reward_infoi)
            # reward, reward_info = tuple(self._agent_reward_merging_reward(_vehicle, _action) for _vehicle, _action in
            #              zip(self.env.controlled_vehicles, self.action))
            # print(reward)
            reward = tuple(reward)
            self.reward_info = reward_info
            return reward
        elif self.reward_type == "exit_reward":
            return tuple(self._agent_reward_exit_reward(_vehicle, _action) for _vehicle, _action in
                         zip(self.env.controlled_vehicles, self.action))

    def _linear_sum_reward(self):
        reward = 0
        if self.reward_type == "type_1":
            reward = sum(self._agent_reward_type_1(_vehicle, _action) for _vehicle, _action in
                         zip(self.env.controlled_vehicles, self.action))
        elif self.reward_type == "type_2":
            reward = sum(self._agent_reward_type_2(_vehicle, _action) for _vehicle, _action in
                         zip(self.env.controlled_vehicles, self.action))
        elif self.reward_type == "type_3":
            reward = sum(self._agent_reward_type_3(_vehicle, _action) for _vehicle, _action in
                         zip(self.env.controlled_vehicles, self.action))

        if self.distance_reward_type == "group":
            positions = []
            for controlled_vehicle in self.env.controlled_vehicles:
                positions.append(controlled_vehicle.position)
            distance = np.amax(positions, axis=0)[0] - np.amin(positions, axis=0)[0]

        reward += self.distance_reward * distance

    # def _agent_reward_type_1(self, vehicle, action):
    #     """
    #            The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
    #            :param action: the last action performed
    #            :return: the corresponding reward
    #            """
    #     neighbours = self.env.road.network.all_side_lanes(vehicle.lane_index)
    #     lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
    #         else vehicle.lane_index[2]
    #     scaled_speed = utils.lmap(vehicle.speed, self.env.config["reward_speed_range"], [0, 1])
    #
    #     # TODO: this must be read from action.py, to determine if it is a lane change or not,
    #     #  for now it's hard-coded to 0 and 2
    #     lane_change = (action == 0 or action == 2)
    #
    #     reward = \
    #         + self.collision_reward * vehicle.crashed \
    #         + self.on_desired_lane_reward * lane / max(len(neighbours) - 1, 1) \
    #         + self.high_speed_reward * np.clip(scaled_speed, 0, 1) \
    #         + self.lane_change_reward * lane_change
    #
    #     if self.normalize_reward:
    #         reward = utils.lmap(reward, [self.collision_reward + self.lane_change_reward, self.high_speed_reward +
    #                                      self.on_desired_lane_reward], [0, 1])
    #
    #     reward = 0 if not vehicle.on_road else reward
    #     return reward

    def _agent_reward_type_1(self, vehicle, action):
        """
               The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
               :param action: the last action performed
               :return: the corresponding reward
               """
        neighbours = self.env.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        scaled_speed = utils.lmap(vehicle.speed, self.reward_speed_range, [0, 1])

        # TODO: this must be read from action.py, to determine if it is a lane change or not,
        #  for now it's hard-coded to 0 and 2
        lane_change = (action == 0 or action == 2)

        reward = \
            + self.collision_reward * vehicle.crashed \
            + self.on_desired_lane_reward * lane / max(len(neighbours) - 1, 1) \
            + self.high_speed_reward * np.clip(scaled_speed, 0, 1) \
            # + self.lane_change_reward * lane_change

        if self.normalize_reward:
            reward = utils.lmap(reward, [self.collision_reward + self.lane_change_reward, self.high_speed_reward +
                                         self.on_desired_lane_reward], [0, 1])

        reward = 0 if not vehicle.on_road else reward
        return reward

    def _agent_reward_type_2(self, vehicle, action):
        """
               The reward is defined to foster driving at high speed, on a given target lane, and to avoid collisions.
               :param action: the last action performed
               :return: the corresponding reward
               """
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        scaled_speed = utils.lmap(vehicle.speed, self.reward_speed_range, [0, 1])

        # TODO: this must be read from action.py, to determine if it is a lane change or not,
        #  for now it's hard-coded to 0 and 2
        lane_change = (action == 0 or action == 2)
        on_lane = (lane == self.target_lane)

        distances = []
        distance = 0
        for v in self.env.controlled_vehicles:
            if v is not vehicle:
                longitudinal_distance = vehicle.front_distance_to(v)
                # distance = np.linalg.norm(v.position - vehicle.position)
                distances.append(longitudinal_distance)
        if self.distance_reward_type == "min":
            distance_to_neighbor_car = min(np.abs(distances))
            distance = utils.lmap(distance_to_neighbor_car,
                                  [Vehicle.LENGTH, self.env.PERCEPTION_DISTANCE], [0, 1])
        elif self.distance_reward_type == "avg":
            average_distance = np.average(np.abs(distances))
            distance = utils.lmap(average_distance,
                                  [Vehicle.LENGTH, self.env.PERCEPTION_DISTANCE], [0, 1])

        reward = \
            + self.collision_reward * vehicle.crashed \
            + self.on_desired_lane_reward * on_lane \
            + self.high_speed_reward * np.clip(scaled_speed, 0, 1) \
            + self.lane_change_reward * lane_change \
            + self.distance_reward * distance

        if self.normalize_reward:
            reward = utils.lmap(reward, [self.collision_reward + self.lane_change_reward, self.high_speed_reward +
                                         self.on_desired_lane_reward], [0, 1])

        reward = 0 if not vehicle.on_road else reward
        return reward

    def _agent_reward_type_3(self, vehicle, action):
        # TODO: this is for following the same lane
        return

    def _agent_reward_merging_reward(self, vehicle, action):
        """
               The reward is to motivate agents to allow the merging human to
                merge and also stay on the desired lane.
               """

        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        scaled_speed = utils.lmap(vehicle.speed, self.reward_speed_range, [0, 1])

        # TODO: this must be read from action.py, to determine if it is a lane change or not,
        #  for now it's hard-coded to 0 and 2

        # TODO: these action values should be read from action class, this can create bugs
        if self.env.config["action"]["action_config"]["lateral"]:
            lane_change = (action == 0 or action == 2)
        else:
            lane_change = False
        on_lane = (lane == self.target_lane)

        distances = []
        distance = 0
        for v in self.env.controlled_vehicles:
            if v is not vehicle:
                longitudinal_distance = vehicle.front_distance_to(v)
                # distance = np.linalg.norm(v.position - vehicle.position)
                distances.append(longitudinal_distance)
        if distances:
            if self.distance_reward_type == "min":
                distance_to_neighbor_car = min(np.abs(distances))
                distance = utils.lmap(distance_to_neighbor_car,
                                      [Vehicle.LENGTH, self.env.PERCEPTION_DISTANCE], [0, 1])
            elif self.distance_reward_type == "avg":
                average_distance = np.average(np.abs(distances))
                distance = utils.lmap(average_distance,
                                      [Vehicle.LENGTH, self.env.PERCEPTION_DISTANCE], [0, 1])

        has_merged, merge_vehicle = self.env.mission_accomplished, self.env.mission_vehicle
        if not merge_vehicle.is_controlled:
            has_merged = has_merged * int(self.sympathy_flag)

        if not self.continuous_mission_reward and has_merged:
            self.env.merged_vehicle = True
            if self.env.merged_counter >= (self.continuous_mission_reward_steps_counter * len(self.env.controlled_vehicles)):
                has_merged = False
            else:
                self.env.merged_counter += 1

        # TODO chooose better names for these variables
        longitudinal_distance = abs(vehicle.front_distance_to(merge_vehicle))
        distance_merge_vehicle = utils.lmap(longitudinal_distance, [Vehicle.LENGTH, self.env.PERCEPTION_DISTANCE],
                                            [0, 1])

        # reward_m_distance =-0.4* np.clip(distance_merge_vehicle, 0,1)
        # TODO fix this distance reward and choose a meaningful name
        reward_m_distance = 0
        # TODO: remove this if not used
        # sum_of_speeds = self.sum_of_speeds(sympathy_flag = self.sympathy_flag , scaled_speed_flag = True)
        # avg_speeds = self.avg_speeds(sympathy_flag=self.sympathy_flag, scaled_speed_flag=True)

        collision_reward_value = self.collision_reward * vehicle.crashed
        on_desired_lane_reward_value = self.on_desired_lane_reward * on_lane
        high_speed_reward_value = self.high_speed_reward * np.clip(scaled_speed, 0, 1)
        lane_change_reward_value = self.lane_change_reward * lane_change
        distance_reward_value = self.distance_reward * np.clip(distance, 0, 1)
        cooperative_reward_value_avg_speeds = self.cooperative_reward_value * \
                                              np.clip(self.avg_speeds, 0, 1)
        cooperative_reward_value_merging_reward = int(self.sympathy_flag) * self.successful_merging_reward * has_merged
        cooperative_reward_value_distance_merge_vehicle = int(self.cooperative_flag)  * \
                                                          self.distance_merged_vehicle_reward * \
                                                          np.clip(distance_merge_vehicle, 0, 1)
        reward = \
            + collision_reward_value \
            + on_desired_lane_reward_value \
            + high_speed_reward_value \
            + lane_change_reward_value \
            + distance_reward_value \
            + cooperative_reward_value_avg_speeds \
            + cooperative_reward_value_merging_reward \
            + cooperative_reward_value_distance_merge_vehicle

        if self.normalize_reward:
            reward = utils.lmap(reward, [self.collision_reward + self.lane_change_reward + self.distance_reward,
                                         self.high_speed_reward + self.on_desired_lane_reward +
                                         int(self.cooperative_flag) * self.distance_merged_vehicle_reward + self.successful_merging_reward * int(self.sympathy_flag) +
                                         self.cooperative_reward_value * int(self.cooperative_flag)], [0, 1])

        reward = 0 if not vehicle.on_road else reward

        reward_info = {"reward_crashed": collision_reward_value,
                       "reward_on_lane": on_desired_lane_reward_value,
                       "reward_scaled_speed": high_speed_reward_value,
                       "reward_lane_change": lane_change_reward_value,
                       "reward_distance": distance_reward_value,
                       "reward_avg_speeds": cooperative_reward_value_avg_speeds,
                       "reward_has_merged": cooperative_reward_value_merging_reward,
                       "reward_distance_merge_vehicle": cooperative_reward_value_distance_merge_vehicle,
                       # "normalized_timestep_reward": reward
                       # "vehilce_id": vehicle.id
                       }

        return reward, reward_info

    def _agent_reward_exit_reward(self, vehicle, action):
        """
               The reward is to motivate agents to allow the merging human to
                merge and also stay on the desired lane.
               """

        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        scaled_speed = utils.lmap(vehicle.speed, self.reward_speed_range, [0, 1])

        # TODO: this must be read from action.py, to determine if it is a lane change or not,
        #  for now it's hard-coded to 0 and 2
        lane_change = (action == 0 or action == 2)
        on_lane = (lane == self.target_lane)

        distances = []
        distance = 0
        for v in self.env.controlled_vehicles:
            if v is not vehicle:
                longitudinal_distance = vehicle.front_distance_to(v)
                # distance = np.linalg.norm(v.position - vehicle.position)
                distances.append(longitudinal_distance)
        if distances:
            if self.distance_reward_type == "min":
                distance_to_neighbor_car = min(np.abs(distances))
                distance = utils.lmap(distance_to_neighbor_car,
                                      [Vehicle.LENGTH, self.env.PERCEPTION_DISTANCE], [0, 1])
            elif self.distance_reward_type == "avg":
                average_distance = np.average(np.abs(distances))
                distance = utils.lmap(average_distance,
                                      [Vehicle.LENGTH, self.env.PERCEPTION_DISTANCE], [0, 1])

        has_exit, exit_vehicle = self.env.mission_accomplished, self.env.mission_vehicle

        if not self.env.config['reward']['continuous_mission_reward'] and has_exit:
            if not self.env.merged_vehicle:
                self.env.merged_vehicle = True
            else:
                has_exit = False

        # TODO chooose better names for these variables
        longitudinal_distance = abs(vehicle.front_distance_to(exit_vehicle))
        distance_merge_vehicle = utils.lmap(longitudinal_distance, [Vehicle.LENGTH, self.env.PERCEPTION_DISTANCE],
                                            [0, 1])

        # reward_m_distance =-0.4* np.clip(distance_merge_vehicle, 0,1)
        # TODO fix this distance reward and choose a meaningful name
        reward_m_distance = 0

        reward = \
            + self.collision_reward * vehicle.crashed \
            + self.on_desired_lane_reward * on_lane \
            + self.high_speed_reward * np.clip(scaled_speed, 0, 1) \
            + self.lane_change_reward * lane_change \
            + self.distance_reward * np.clip(distance, 0, 1) \
            + self.cooperative_reward_value * np.clip(self.avg_speeds, 0, 1) \
            + self.cooperative_reward_value *  self.successful_merging_reward * has_exit \
            + self.cooperative_reward_value *  self.distance_merged_vehicle_reward * np.clip(distance_merge_vehicle, 0, 1)

        if self.normalize_reward:
            reward = utils.lmap(reward, [self.collision_reward + self.lane_change_reward + self.distance_reward,
                                         self.high_speed_reward +
                                         self.on_desired_lane_reward + self.successful_merging_reward], [-1, 1])

        reward = 0 if not vehicle.on_road else reward
        return reward

    def check_merging_human(self):
        # TODO
        vehicle = None
        for vehicle in self.env.road.vehicles:
            if vehicle.id == self.env.config['merging_vehicle']['id']:
                _from, _to, _id = vehicle.target_lane_index
                right_lane = len(self.env.road.network.graph['b']['c']) - 1
                if _from == 'b' and _to == 'c' and _id != right_lane and self.env.merged_vehicle == False:
                    if self.env.config['reward']['continuous_mission_reward']:
                        self.env.merged_vehicle = True
                    return True, vehicle
                else:
                    return False, vehicle
