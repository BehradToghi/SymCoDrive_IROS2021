import time
import csv
import os
import numpy as np
import copy
class LogCreator():

    RAW_LOG_FOLDER = 'raw_logfiles'
    TIMESTEP_LOG_FOLDER = 'timestep_logs'
    EPISODE_LOGFILE = 'episode_logfile'
    TIMESTEP_LOGFILE = 'timestep_logfile'

    EPISODE_FIELD_NAMES = ['episode', 'episode_reward', 'episode_length', 'episode_average_speed_all',
                           'episode_average_speed_controlled', 'episode_average_speed_human',
                           'episode_average_distance_all', 'episode_average_distance_controlled',
                           'episode_average_distance_human', 'mission_time']

    EPISODE_INDIVIDUAL_FIELD_NAMES = ['episode', 'vehicle_id', 'vehicle_is_controlled', 'episode_reward',
                                      'episode_length', 'vehicle_average_speed', 'vehicle_average_distance',
                                      'mission_time']

    EPISODE_MISSION_FIELD_NAMES = ['episode', 'vehicle_id', 'vehicle_is_controlled', 'episode_reward',
                                   'episode_length', 'vehicle_average_speed', 'vehicle_average_distance',
                                   'mission_time']

    # common field or different ? TIMESTEP_FIELD_NAMES_CONTROLLED ?
    # TIMESTEP_FIELD_NAMES = ['timestep', 'is_controlled', 'vehicle_id', 'timestep_reward', 'vehicle_speed',
    #                         'vehicle_distance', 'mission_accomplished']
    TIMESTEP_FIELD_NAMES = ['timestep', 'is_controlled', 'vehicle_id', 'timestep_reward', 'vehicle_speed',
                            'vehicle_distance', 'mission_accomplished']


    def __init__(self, evaluation):

        self.evaluation = evaluation
        self.run_directory = self.evaluation.run_directory
        self.controlled_vehicles_count = len(self.evaluation.env.controlled_vehicles)
        self.vehicles_count = len(self.evaluation.env.road.vehicles)
        self.humans_count = self.vehicles_count - self.controlled_vehicles_count
        #TODO
        # self.mission_vehicle_id = self.evaluation.env.config['scenario']['mission_vehicle_id']
        self.mission_vehicle_id = -1
        self.mission_time = None

        self.average_episode_logfile_name = self.get_logfile_name('episode_average')
        self.create_raw_log_folder()
        self.rewards_keys = []
        self.rewards_keys_episode = []
        self.update_field_once = 1
        self.TIMESTEP_FIELD_NAMES_EXTRA = copy.deepcopy(self.TIMESTEP_FIELD_NAMES)

    def create_raw_log_folder(self):
        log_folder_path = os.path.join(self.run_directory, self.RAW_LOG_FOLDER)
        if not os.path.exists(log_folder_path):
            os.makedirs(log_folder_path)

    def create_timestep_log_folder(self, episode, vehicle_id):

        vehicle_folder_name = "vehicle_" + str(vehicle_id)
        log_folder_path = os.path.join(self.run_directory, self.RAW_LOG_FOLDER, self.TIMESTEP_LOG_FOLDER,
                                    vehicle_folder_name)
        if not os.path.exists(log_folder_path):
            os.makedirs(log_folder_path)

    def create_episode_logfiles(self):

        with open(self.average_episode_logfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.EPISODE_FIELD_NAMES)
            writer.writeheader()

    def get_logfile_name(self, log_type, **kwargs):
        vehicle_id = str(kwargs.get('vehicle_id', 0))

        assert (log_type == 'episode_average' or log_type == 'episode_mission' or
                log_type == 'episode_individual' or log_type == 'timestep'), \
            "'get_logfile_name()' only accepts 'episode_average' or 'timestep' as 'log_type'"
        logfile_name = None

        if log_type == 'episode_average':
            logfile_name = self.EPISODE_LOGFILE  + '_average.csv'
        elif log_type == 'episode_individual':
            logfile_name = self.EPISODE_LOGFILE + '_individual_' + vehicle_id + '.csv'
        elif log_type == 'episode_mission':
            logfile_name = self.EPISODE_LOGFILE + '_mission' + '.csv'
        elif log_type == 'timestep':
            timestep = str(kwargs.get('timestep', 0))
            episode = str(kwargs.get('episode', 0))
            logfile_name = self.TIMESTEP_LOGFILE + '_vehicle_' + vehicle_id + '_episode_' + episode + '.csv'
            vehicle_folder_name = "vehicle_" + str(vehicle_id)
            logfile_name = os.path.join(self.TIMESTEP_LOG_FOLDER,
                                        vehicle_folder_name, logfile_name)

        logfile_path = os.path.join(self.run_directory, self.RAW_LOG_FOLDER, logfile_name)

        return logfile_path

    def episode_info_logger(self, episode):
        start_time = time.time()

        ############# Calculations
        rewards_averaged_over_agents = self.evaluation.rewards_averaged_over_agents
        rewards_individual_agents = self.evaluation.rewards_individual_agents

        reward_total_episode = sum(rewards_averaged_over_agents)
        episode_length = self.evaluation.episode_length

        episode_info = self.evaluation.episode_info

        # speed calculations
        # this keeps the sum of speeds over timesteps of an episode for all vehicles
        speeds_container = np.zeros(self.vehicles_count)

        # this keeps the sum of rewards components over timesteps of an episode for all controlled vehicles
        reward_components_length = len(episode_info[0]["reward_info"][0])
        rewards_container = np.zeros((self.controlled_vehicles_count,reward_components_length))

        # distance calculations
        # keeps the sum over timesteps
        distances_container = np.zeros(self.vehicles_count)
        # counts how many non-None distances have occured for each vehicle
        distances_counter = np.zeros(self.vehicles_count)

        if self.update_field_once:
            self.rewards_keys = list(episode_info[0]["reward_info"][0].keys())
            self.rewards_keys_episode = ["episode_average_" + char for char in self.rewards_keys]

            self.TIMESTEP_FIELD_NAMES_EXTRA.extend(self.rewards_keys)
            self.EPISODE_FIELD_NAMES.extend(self.rewards_keys_episode)
            if len(episode_info[0]["other_vehicle_info_debug"]) > 0:
                self.TIMESTEP_FIELD_NAMES_EXTRA.extend(list(episode_info[0]["other_vehicle_info_debug"][0].keys()))

            self.EPISODE_INDIVIDUAL_FIELD_NAMES.extend(self.rewards_keys)
            self.EPISODE_MISSION_FIELD_NAMES.extend(self.rewards_keys)
            self.create_episode_logfiles()

            self.update_field_once = 0

        # -1 means mission was never accomplished
        self.mission_time = -1
        for step in range(episode_length):
            # TODO: this is currently only for merging but should be general
            info_at_timestep = episode_info[step]
            timestep = info_at_timestep['timestep']

            rewards_at_timestep = rewards_individual_agents[step]
            speeds_container = np.add(speeds_container, info_at_timestep['vehicle_speeds'])
            reward_values = [np.array(list(rewards.values())) for rewards in info_at_timestep["reward_info"]]
            reward_values = np.array(reward_values)
            rewards_container = np.add(rewards_container, reward_values)
            for i, distance in enumerate(info_at_timestep['vehicle_distances']):
                if not distance == None:
                    distances_counter[i] += 1
                    distances_container[i] += distance

            # checking if the goal is accomplished
            if (info_at_timestep['mission_accomplished'] and self.mission_time == -1):
                self.mission_time = timestep

            # creating timestep logs
            if self.evaluation.create_timestep_log:
                vehicle_ids = info_at_timestep['vehicle_ids']

                for i, vehicle_id in enumerate (vehicle_ids):
                    self.create_timestep_log_folder(episode, vehicle_id)

                    vehicle_timestep_reward = 0
                    if vehicle_id in info_at_timestep['reward_ids']:
                        j = np.where(np.array(info_at_timestep['reward_ids'])==vehicle_id)[0][0]
                        vehicle_timestep_reward = rewards_at_timestep[j]

                    individual_timestep_log = {
                                            'timestep': timestep,
                                            'is_controlled': info_at_timestep['vehicle_is_controlled'][i],
                                            'vehicle_id': vehicle_id,
                                            'timestep_reward': vehicle_timestep_reward,
                                            'vehicle_speed': info_at_timestep['vehicle_speeds'][i],
                                            'vehicle_distance': info_at_timestep['vehicle_distances'][i],
                                            'mission_accomplished': info_at_timestep['mission_accomplished']}

                    individual_timestep_log_name = self.get_logfile_name('timestep', episode=episode,
                                                                         vehicle_id=vehicle_id, timestep=timestep)
                    with open(individual_timestep_log_name, 'a') as csvfile:
                        if vehicle_id in info_at_timestep['reward_ids']:
                            # writer = csv.DictWriter(csvfile, fieldnames=self.TIMESTEP_FIELD_NAMES_CONTROLLED)
                            j = info_at_timestep['reward_ids'].index(vehicle_id)
                            individual_timestep_log.update(info_at_timestep["reward_info"][j])

                        if len(episode_info[0]["other_vehicle_info_debug"]) > 0:
                            individual_timestep_log.update(info_at_timestep["other_vehicle_info_debug"][i])
                        # else:
                        #     writer = csv.DictWriter(csvfile, fieldnames=self.TIMESTEP_FIELD_NAMES)
                        writer = csv.DictWriter(csvfile, fieldnames=self.TIMESTEP_FIELD_NAMES_EXTRA)
                        if timestep == 1:
                            writer.writeheader()
                        writer.writerow(individual_timestep_log)

        ### Calculating average values (averaged over the timesteps of an episode)
        mask = episode_info[0]['vehicle_is_controlled']
        ## Speeds
        # for all vehicles separately
        vehicles_average_speeds = speeds_container / episode_length
        controlled_average_speeds = np.multiply(vehicles_average_speeds, mask)
        human_average_speeds = vehicles_average_speeds - controlled_average_speeds
        # averaged over all vehicles
        episode_average_speed_all = sum(vehicles_average_speeds) / self.vehicles_count
        episode_average_speed_controlled = sum(controlled_average_speeds) / self.controlled_vehicles_count
        episode_average_speed_human = sum(human_average_speeds) / self.humans_count

        ## Distances
        # here we remove the entries that have a distance_counter==0 because that means they never had a vehicle in
        # front of them and hence should not be considered in the averaging
        no_distance_indices = np.where(distances_counter==0)

        distances_counter_masked = np.delete(distances_counter, no_distance_indices)
        distances_container_masked = np.delete(distances_container, no_distance_indices)
        mask = np.delete(mask, no_distance_indices)


        vehicles_average_distances = 0 if not distances_counter_masked else distances_container_masked / distances_counter_masked
        controlled_average_distances = np.delete(vehicles_average_distances,   np.argwhere(1*np.logical_not(mask)))
        human_average_distances = np.delete(vehicles_average_distances, np.argwhere(mask))

        # averaged over all vehicles
        episode_average_distance_all = 0 if not vehicles_average_distances else  np.average(vehicles_average_distances)
        episode_average_distance_controlled = 0 if not controlled_average_distances else  np.average(controlled_average_distances)
        episode_average_distance_human = 0 if not human_average_distances else  np.average(human_average_distances)

        rewards_container_average = rewards_container/ episode_length
        episode_rewards_components_average = np.average(rewards_container_average,axis=0)
        # average over all controlled vehicles reward components

        episode_average_reward_log = {self.rewards_keys_episode[i]: episode_rewards_components_average[i] for i in range(0,len(self.rewards_keys_episode))}
        episode_average_log = {'episode': episode,
                               'episode_reward': reward_total_episode,
                               'episode_length': episode_length,
                               'episode_average_speed_all': episode_average_speed_all,
                               'episode_average_speed_controlled': episode_average_speed_controlled,
                               'episode_average_speed_human': episode_average_speed_human,
                               'episode_average_distance_all': episode_average_distance_all,
                               'episode_average_distance_controlled': episode_average_distance_controlled,
                               'episode_average_distance_human': episode_average_distance_human,
                               'mission_time': self.mission_time}

        episode_average_log.update(episode_average_reward_log)

        with open(self.average_episode_logfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.EPISODE_FIELD_NAMES)
            writer.writerow(episode_average_log)

        #### Individual Logs
        vehicle_ids = np.array(episode_info[0]['vehicle_ids'])
        vehicle_is_controlled_arr = np.array(episode_info[0]['vehicle_is_controlled'])
        vehicle_reward_ids = np.array(episode_info[0]['reward_ids'])

        if (self.evaluation.individual_episode_log_level == 2 or self.evaluation.individual_episode_log_level==3):
            vehicle_id = None
            vehicle_is_controlled = None
            vehicle_reward = None
            vehicle_average_speed = None

            controlled_indices = np.argwhere(vehicle_is_controlled_arr)
            vehicle_rewards = np.sum(rewards_individual_agents, axis=0)
            for i in controlled_indices:
                i = i[0]
                vehicle_id = vehicle_ids[i]
                if vehicle_id == self.mission_vehicle_id:
                    continue

                # TODO: here check if it's controlled, if not reward = None
                vehicle_reward_index = np.where(vehicle_reward_ids == vehicle_id)[0][0]
                # vehicle_reward_indexv = episode_info[0]['reward_ids'].index(vehicle_id)
                vehicle_reward = vehicle_rewards[vehicle_reward_index]

                vehicle_average_speed =  vehicles_average_speeds[i]
                vehicle_is_controlled = vehicle_is_controlled_arr[i]

                vehicle_average_distance = float("inf")
                if i not in no_distance_indices[0]:
                    vehicle_average_distance = distances_container[i] / distances_counter[i]

                rewards_container_average_vehicle = rewards_container_average[vehicle_reward_index,:]
                episode_individual_reward_log = {self.rewards_keys[j]: rewards_container_average_vehicle[j] for j
                                              in range(0, len(self.rewards_keys))}

                episode_individual_log = {'episode': episode,
                                          'vehicle_id': vehicle_id,
                                          'vehicle_is_controlled': vehicle_is_controlled,
                                          'episode_reward': vehicle_reward,
                                          'episode_length': episode_length,
                                          'vehicle_average_speed': vehicle_average_speed,
                                          'vehicle_average_distance': vehicle_average_distance,
                                          'mission_time': self.mission_time}

                episode_individual_log.update(episode_individual_reward_log)

                individual_log_name = self.get_logfile_name('episode_individual', vehicle_id=vehicle_id)
                with open(individual_log_name, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.EPISODE_INDIVIDUAL_FIELD_NAMES)
                    if episode == 1:
                        writer.writeheader()
                    writer.writerow(episode_individual_log)

        ### Log for mission vehicle
        mission_vehicle_index = np.where(vehicle_ids == self.mission_vehicle_id)[0][0]

        mission_vehicle_average_speed = vehicles_average_speeds[mission_vehicle_index]
        mission_vehicle_is_controlled = vehicle_is_controlled_arr[mission_vehicle_index]
        mission_vehicle_average_distance = float("inf")
        if mission_vehicle_index not in no_distance_indices[0]:
            mission_vehicle_average_distance = distances_container[mission_vehicle_index] \
                                               / distances_counter[mission_vehicle_index]

        mission_vehicle_reward = None
        if mission_vehicle_is_controlled:
            vehicle_rewards = np.sum(rewards_individual_agents, axis=0)
            mission_reward_index = np.where(vehicle_reward_ids == self.mission_vehicle_id)[0][0]
            mission_vehicle_reward = vehicle_rewards[mission_reward_index]



        episode_mission_log = {'episode': episode,
                                  'vehicle_id': self.mission_vehicle_id,
                                  'vehicle_is_controlled': mission_vehicle_is_controlled,
                                  'episode_reward': mission_vehicle_reward,
                                  'episode_length': episode_length,
                                  'vehicle_average_speed': mission_vehicle_average_speed,
                                  'vehicle_average_distance': mission_vehicle_average_distance,
                                  'mission_time': self.mission_time}

        if self.mission_vehicle_id in episode_info[0]['reward_ids'] :
            mission_vehicle_reward_index = episode_info[0]['reward_ids'].index(self.mission_vehicle_id )
            rewards_container_average_vehicle = rewards_container_average[mission_vehicle_reward_index, :]
            episode_mission_reward_log = {self.rewards_keys[j]: rewards_container_average_vehicle[j] for j
                                             in range(0, len(self.rewards_keys))}

            episode_mission_log.update(episode_mission_reward_log)

        individual_log_name = self.get_logfile_name('episode_mission')
        with open(individual_log_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.EPISODE_MISSION_FIELD_NAMES)
            if episode == 1:
                writer.writeheader()
            writer.writerow(episode_mission_log)

        logging_time = time.time() - start_time
        # print(">>>>>>>>>>>>>> LOG FILE SUCCESSFULLY UPDATED IN  = {:5f} ms".format(1000*logging_time))
        return episode_average_log