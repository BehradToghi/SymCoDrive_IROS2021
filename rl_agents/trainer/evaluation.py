import datetime
import json
import logging
import os
from multiprocessing.pool import Pool
from pathlib import Path
import numpy as np
from tensorboardX import SummaryWriter
import time

import rl_agents.trainer.logger
from rl_agents.agents.common.factory import load_environment, load_agent
from rl_agents.agents.common.graphics import AgentGraphics
from rl_agents.agents.common.memory import Transition
from rl_agents.utils import near_split, zip_with_singletons
from rl_agents.configuration import serialize
from rl_agents.trainer.graphics import RewardViewer
from rl_agents.trainer.monitor import MonitorV2
from rl_agents.trainer.log_creator import LogCreator

logger = logging.getLogger(__name__)

# TODO: clean this up
import cv2
# from getkey import getkey
from sys import exit


#
# import keyboard

# import sys
# import tty
# import termios
# from time import sleep
# def getch():
#
#     fd = sys.stdin.fileno()
#     old_settings = termios.tcgetattr(fd)
#
#     try:
#         tty.setraw(sys.stdin.fileno())
#         ch = sys.stdin.read(1)
#     finally:
#         termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#
#     return ch

class Evaluation(object):
    """
        The evaluation of an agent interacting with an environment to maximize its expected reward.
    """

    OUTPUT_FOLDER = 'out2'
    SAVED_MODELS_FOLDER = 'saved_models'
    RUN_FOLDER = 'run_{}_{}'
    METADATA_FILE = 'metadata.{}.json'
    LOGGING_FILE = 'logging.{}.log'

    def __init__(self,
                 env,
                 agent,
                 directory=None,
                 run_directory=None,
                 num_episodes=1000,
                 training=True,
                 test=False,
                 sim_seed=None,
                 recover=None,
                 display_env=True,
                 display_agent=True,
                 display_rewards=True,
                 close_env=True,
                 create_episode_log=False,
                 individual_episode_log_level=0,
                 create_timestep_log=False,
                 timestep_log_freq=1,
                 individual_reward_tensorboard=False,
                 timing=False,
                 options=None):
        """

        :param env: The environment to be solved, possibly wrapping an AbstractEnv environment
        :param AbstractAgent agent: The agent solving the environment
        :param Path directory: Workspace directory path
        :param Path run_directory: Run directory path
        :param int num_episodes: Number of episodes run
        !param training: Whether the agent is being trained or tested
        :param sim_seed: The seed used for the environment/agent randomness source
        :param recover: Recover the agent parameters from a file.
                        - If True, it the default latest save will be used.
                        - If a string, it will be used as a path.
        :param display_env: Render the environment, and have a monitor recording its videos
        :param display_agent: Add the agent graphics to the environment viewer, if supported
        :param display_rewards: Display the performances of the agent through the episodes
        :param close_env: Should the environment be closed when the evaluation is closed

        """
        self.env = env
        self.env.training = training
        self.env.options = options
        self.agent = agent
        self.num_episodes = num_episodes
        self.training = training
        self.test_flag = test
        self.sim_seed = sim_seed
        self.close_env = close_env
        self.display_env = display_env
        # TODO Rodo
        self.options = options
        self.timing = timing
        self.directory = Path(directory or self.default_directory)
        exp_json = options["--environment"].split('/')[-1]

        default_run_directory = self.default_run_directory + "_" + exp_json.split('.')[0]
        tmp = default_run_directory
        if training:
            default_run_directory = os.path.join("train", tmp)
        if test:
            default_run_directory = os.path.join("test", tmp + "-test")
        self.run_directory = self.directory / (run_directory or default_run_directory)
        self.monitor = MonitorV2(env,
                                 self.run_directory,
                                 video_callable=(None if self.display_env else False), options=options)
        self.writer = SummaryWriter(str(self.run_directory))
        self.agent.set_writer(self.writer)
        self.write_logging()
        self.write_metadata()
        self.filtered_agent_stats = 0
        self.best_agent_stats = -np.infty, 0

        # Declaring logging variables
        self.rewards_individual_agents = None
        self.rewards_averaged_over_agents = None
        self.episode_length = None
        self.episode_info = None

        # To calculate the episode ET
        self.episode_start_time = 0

        self.recover = recover
        if self.recover:
            self.load_agent_model(self.recover)

        self.create_episode_log = create_episode_log
        self.individual_episode_log_level = int(individual_episode_log_level)
        self.create_timestep_log = create_timestep_log
        self.timestep_log_freq = int(timestep_log_freq)
        self.individual_reward_tensorboard = individual_reward_tensorboard
        self.log_creator = None

        if display_agent:
            try:
                # Render the agent within the environment viewer, if supported
                self.env.render(mode='rgb_array')
                # TODO
                self.env.unwrapped.viewer.set_agent_display(
                    lambda agent_surface, sim_surface: AgentGraphics.display(self.agent, agent_surface, sim_surface))
                self.env.unwrapped.viewer.directory = self.run_directory
            except AttributeError:
                logger.info("The environment viewer doesn't support agent rendering.")
        self.reward_viewer = None
        if display_rewards:
            self.reward_viewer = RewardViewer()
        self.observation = None

    def train(self):
        self.training = True
        self.env.training = True
        if getattr(self.agent, "batched", False):
            self.run_batched_episodes()
        else:
            self.run_episodes()
        self.close()

    def test(self):
        """
        Test the agent.

        If applicable, the agent model should be loaded before using the recover option.
        """
        self.training = False
        self.env.training = False
        # always show when test
        if self.display_env:
            self.monitor.video_callable = MonitorV2.always_call_video
        try:
            self.agent.eval()
        except AttributeError:
            pass

        self.run_episodes()
        self.close()

    def run_episodes(self):
        if (self.create_episode_log or self.create_episode_log):
            self.log_creator = LogCreator(self)

        for episode in range(self.num_episodes):
            if self.test_flag and (self.num_episodes - episode) < 50:
                self.create_timestep_log = True
                self.monitor.options['--video_save_freq'] = 1

            self.episode_start_time = time.time()
            # Run episode
            terminal = False
            self.seed(episode)
            if self.timing:
                reset_start_time = time.time()
            self.reset()

            if self.timing:
                reset_elapsed_time = 1000 * (time.time() - reset_start_time)
                print("Reset scenario time + {:.1f}ms".format(reset_elapsed_time))

            # Resetting episode variables
            self.rewards_averaged_over_agents = []
            self.rewards_individual_agents = []
            self.episode_length = 0
            self.episode_info = []

            steps = 0
            break_flag = False
            while not terminal:
                # Step until a terminal step is reached
                if self.timing:
                    print("INFO: Start step")
                    step_start_time = time.time()
                reward, terminal, info = self.step()
                # print("episode: ", episode, "steps: ", steps)
                if self.timing:
                    step_elapsed_time = 1000 * (time.time() - step_start_time)
                    print("INFO: Step time + {:.1f}ms".format(step_elapsed_time))

                # Behrad: I am changing this so we can keep the tuple reward and access each agent's reward.
                if isinstance(reward, tuple):
                    self.rewards_averaged_over_agents.append(sum(reward) / len(reward))
                else:
                    self.rewards_averaged_over_agents.append(reward)
                self.rewards_individual_agents.append(reward)

                # Keeping track of info, each info dictionary that step() returns includes information
                # for all controlled vehicles
                # this is a list of dictionaries, length of the list should be equal to the length of episode
                self.episode_info.append(info)

                # Catch interruptions
                try:
                    if self.env.unwrapped.done:
                        break
                except AttributeError:
                    pass
                steps += 1
                # TODO: clean this up, do we need this?
                #
                # Rodo
                # k = cv2.waitKey(1) & 0xFF
                # if k == ord('q') or k == ord('Q'):
                #     break_flag = True
                #     break

                # ch = getch()
                # ch = getkey(blocking=False)
                # if ch == "q" or ch=='Q':
                # if keyboard.is_pressed('q'):  # You must be root to use this library on linux
                #     break_flag=True
                #     break

            # End of episode
            self.after_all_episodes(episode + 1, info)
            self.after_some_episodes(episode + 1)

            # press 'q' to exit  keypress to interrupt the training and save the latest model and exit training
            if break_flag:
                # save the latest model and exit training
                if self.training:
                    self.save_agent_model("break_episode_" + str(episode))
                break

        # Rodo
        # print("End of training")

    def step(self):
        """
            Plan a sequence of actions according to the agent policy, and step the environment accordingly.
        """
        # Query agent for actions sequence

        if self.timing:
            predict_start_time = time.time()
        actions = self.agent.plan(self.observation)
        if self.timing:
            predict_elapsed_time = 1000 * (time.time() - predict_start_time)
            print("INFO: --Action Foward Prediction time + {:.1f}ms".format(predict_elapsed_time))

        if not actions:
            raise Exception("The agent did not plan any action")

        # Forward the actions to the environment viewer
        try:
            self.env.unwrapped.viewer.set_agent_action_sequence(actions)
        except AttributeError:
            pass

        # Step the environment
        previous_observation, action = self.observation, actions[0]

        if self.timing:
            print("INFO: --Start Monitor step")
            step_start_time = time.time()
        self.observation, reward, terminal, info = self.monitor.step(action)
        if self.timing:
            step_elapsed_time = 1000 * (time.time() - step_start_time)
            print("INFO: --Monitor Step time + {:.1f}ms".format(step_elapsed_time))

        # Record the experience.
        try:
            if self.timing:
                train_start_time = time.time()

            return_info_agent = self.agent.record(previous_observation, action, reward, self.observation, terminal, info)
            if self.timing:
                train_elapsed_time = 1000 * (time.time() - train_start_time)
                print("INFO: --Record and train time + {:.1f}ms".format(train_elapsed_time))


        except NotImplementedError:
            pass
        if return_info_agent:
            info["return_info_agent"] = return_info_agent
        else:
            info["return_info_agent"] = [0, 0]
        return reward, terminal, info

    def run_batched_episodes(self):
        """
            Alternatively,
            - run multiple sample-collection jobs in parallel
            - update model
        """
        episode = 0
        episode_duration = 14  # TODO: use a fixed number of samples instead
        batch_sizes = near_split(self.num_episodes * episode_duration, size_bins=self.agent.config["batch_size"])
        self.agent.reset()
        for batch, batch_size in enumerate(batch_sizes):
            logger.info("[BATCH={}/{}]---------------------------------------".format(batch + 1, len(batch_sizes)))
            logger.info("[BATCH={}/{}][run_batched_episodes] #samples={}".format(batch + 1, len(batch_sizes),
                                                                                 len(self.agent.memory)))
            logger.info("[BATCH={}/{}]---------------------------------------".format(batch + 1, len(batch_sizes)))
            # Save current agent
            model_path = self.save_agent_model(identifier=batch)

            # Prepare workers
            env_config, agent_config = serialize(self.env), serialize(self.agent)
            cpu_processes = self.agent.config["processes"] or os.cpu_count()
            workers_sample_counts = near_split(batch_size, cpu_processes)
            workers_starts = list(np.cumsum(np.insert(workers_sample_counts[:-1], 0, 0)) + np.sum(batch_sizes[:batch]))
            base_seed = self.seed(batch * cpu_processes)[0]
            workers_seeds = [base_seed + i for i in range(cpu_processes)]
            workers_params = list(zip_with_singletons(env_config,
                                                      agent_config,
                                                      workers_sample_counts,
                                                      workers_starts,
                                                      workers_seeds,
                                                      model_path,
                                                      batch))

            # Collect trajectories
            logger.info("Collecting {} samples with {} workers...".format(batch_size, cpu_processes))
            if cpu_processes == 1:
                results = [Evaluation.collect_samples(*workers_params[0])]
            else:
                with Pool(processes=cpu_processes) as pool:
                    results = pool.starmap(Evaluation.collect_samples, workers_params)
            trajectories = [trajectory for worker in results for trajectory in worker]

            # Fill memory
            for trajectory in trajectories:
                if trajectory[-1].terminal:  # Check whether the episode was properly finished before logging
                    self.rewards_averaged_over_agents = [transition.reward for transition in trajectory]
                    self.after_all_episodes(episode)
                episode += 1
                [self.agent.record(*transition) for transition in trajectory]

            # Fit model
            self.agent.update()

    @staticmethod
    def collect_samples(environment_config, agent_config, count, start_time, seed, model_path, batch):
        """
            Collect interaction samples of an agent / environment pair.

            Note that the last episode may not terminate, when enough samples have been collected.

        :param dict environment_config: the environment configuration
        :param dict agent_config: the agent configuration
        :param int count: number of samples to collect
        :param start_time: the initial local time of the agent
        :param seed: the env/agent seed
        :param model_path: the path to load the agent model from
        :param batch: index of the current batch
        :return: a list of trajectories, i.e. lists of Transitions
        """
        env = load_environment(environment_config)
        env.seed(seed)

        if batch == 0:  # Force pure exploration during first batch
            agent_config["exploration"]["final_temperature"] = 1
        agent_config["device"] = "cpu"
        agent = load_agent(agent_config, env)
        agent.load(model_path)
        agent.seed(seed)
        agent.set_time(start_time)

        state = env.reset()
        episodes = []
        trajectory = []
        for _ in range(count):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            trajectory.append(Transition(state, action, reward, next_state, done, info))
            if done:
                state = env.reset()
                episodes.append(trajectory)
                trajectory = []
            else:
                state = next_state
        if trajectory:  # Unfinished episode
            episodes.append(trajectory)
        env.close()
        return episodes

    def save_agent_model(self, identifier, do_save=True):
        # Create the folder if it doesn't exist
        permanent_folder = self.directory / self.SAVED_MODELS_FOLDER
        os.makedirs(permanent_folder, exist_ok=True)

        episode_path = None
        if do_save:
            episode_path = Path(self.monitor.directory) / "checkpoint-{}.tar".format(identifier)
            try:
                self.agent.save(filename=permanent_folder / "latest.tar")
                episode_path = self.agent.save(filename=episode_path)
                if episode_path:
                    logger.info("Saved {} model to {}".format(self.agent.__class__.__name__, episode_path))
            except NotImplementedError:
                pass
        return episode_path

    def load_agent_model(self, model_path):
        if model_path is True:
            # model_path = self.directory / self.SAVED_MODELS_FOLDER / "latest.tar"
            model_path = self.directory / self.SAVED_MODELS_FOLDER / "model.tar"
        if isinstance(model_path, str):
            model_path = Path(model_path)
            if not model_path.exists():
                model_path = self.directory / self.SAVED_MODELS_FOLDER / model_path
        try:
            model_path = self.agent.load(filename=model_path)
            if model_path:
                logger.info("Loaded {} model from {}".format(self.agent.__class__.__name__, model_path))
        except FileNotFoundError:
            logger.warning("No pre-trained model found at the desired location.")
        except NotImplementedError:
            pass

    def after_all_episodes(self, episode, info=None):

        rewards_individual_agents = np.array(self.rewards_individual_agents)
        rewards_averaged_over_agents = np.array(self.rewards_averaged_over_agents)
        self.episode_length = rewards_individual_agents.shape[0]
        if len(rewards_individual_agents.shape) > 1:
            controlled_vehicle_count = rewards_individual_agents.shape[1]
        else:
            controlled_vehicle_count = 1

        assert controlled_vehicle_count == len(self.env.controlled_vehicles), \
            "Length of each row in reward should be equal to the number of controlled vehicles"
        gamma = self.agent.config.get("gamma", 1)
        self.writer.add_scalar('episode/length', self.episode_length, episode)
        reward_total_episode = sum(rewards_averaged_over_agents)
        self.writer.add_scalar('episode/total_reward', reward_total_episode, episode)

        if self.individual_reward_tensorboard:
            # logging individual rewards for each controlled_vehicle
            individual_rewards_dict = {}
            individual_rewards_title = f'individual_stats/agent_rewards'
            for n in range(controlled_vehicle_count):
                agent_name = 'agent' + str(n + 1)
                agent_reward_array = sum(rewards_individual_agents[:, n])
                individual_rewards_dict[agent_name] = agent_reward_array
            self.writer.add_scalars(individual_rewards_title, individual_rewards_dict, episode)

        self.writer.add_scalar('episode/return',
                               sum(r * gamma ** t for t, r in enumerate(rewards_averaged_over_agents)), episode)
        self.writer.add_histogram('episode/rewards', rewards_averaged_over_agents, episode)

        # Create raw logfiles
        if self.create_episode_log:
            logged_info = self.log_creator.episode_info_logger(episode)
            # Adding logged info to TensorBoard
            self.writer.add_scalar('episode/mission_time', logged_info['mission_time'], episode)

            self.writer.add_scalar('episode_average_speeds/episode_average_speed_all',
                                   logged_info['episode_average_speed_all'], episode)
            self.writer.add_scalar('episode_average_speeds/episode_average_speed_controlled',
                                   logged_info['episode_average_speed_controlled'], episode)
            self.writer.add_scalar('episode_average_speeds/episode_average_speed_human',
                                   logged_info['episode_average_speed_human'], episode)

            self.writer.add_scalar('episode_average_distances/episode_average_distance_all',
                                   logged_info['episode_average_distance_all'], episode)
            self.writer.add_scalar('episode_average_distances/episode_average_distance_controlled',
                                   logged_info['episode_average_distance_controlled'], episode)
            self.writer.add_scalar('episode_average_distances/episode_average_distance_human',
                                   logged_info['episode_average_distance_human'], episode)

        # Calculate episode ET in ms
        episode_elapsed_time = 1000 * (time.time() - self.episode_start_time)
        if info:
            logger.info("Episode {} done in {:.1f}ms - step duration: {},-  episode duration: {}, total episode reward: {:.1f}, replay memory: #{:n}, {:.3f}MB".
                        format(episode, episode_elapsed_time,episode_elapsed_time/self.episode_length, self.episode_length,
                               reward_total_episode, info["return_info_agent"][0], info["return_info_agent"][1]))
        else:
            logger.info("Episode {} done in {:.1f}ms - episode duration: {}, total episode reward: {:.1f}".
                        format(episode, episode_elapsed_time, self.episode_length,
                               reward_total_episode))

    def after_some_episodes(self, episode, best_increase=1.1, episodes_window=50):

        rewards = self.rewards_averaged_over_agents
        # TODO Rodo
        if self.options == None:
            if self.monitor.is_episode_selected():
                # Save the model
                if self.training:
                    self.save_agent_model(episode)
        else:
            if self.monitor.is_episode_selected_for_modelsave():
                # Save the model
                if self.training:
                    self.save_agent_model(episode)

        if self.training:
            # Save best model so far, averaged on a window
            best_reward, best_episode = self.best_agent_stats
            self.filtered_agent_stats += 1 / episodes_window * (np.sum(rewards) - self.filtered_agent_stats)
            if self.filtered_agent_stats > best_increase * best_reward \
                    and episode >= best_episode + episodes_window:
                self.best_agent_stats = (self.filtered_agent_stats, episode)
                self.save_agent_model("best")

    @property
    def default_directory(self):
        return Path(self.OUTPUT_FOLDER) / self.env.unwrapped.__class__.__name__ / self.agent.__class__.__name__

    @property
    def default_run_directory(self):
        return self.RUN_FOLDER.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid())

    def write_metadata(self):
        metadata = dict(env=serialize(self.env), agent=serialize(self.agent))
        file_infix = '{}.{}'.format(self.monitor.monitor_id, os.getpid())
        file = self.run_directory / self.METADATA_FILE.format(file_infix)
        with file.open('w') as f:
            json.dump(metadata, f, sort_keys=True, indent=4)

    def write_logging(self):
        file_infix = '{}.{}'.format(self.monitor.monitor_id, os.getpid())
        rl_agents.trainer.logger.configure()
        rl_agents.trainer.logger.add_file_handler(self.run_directory / self.LOGGING_FILE.format(file_infix))

    def seed(self, episode=0):
        seed = self.sim_seed + episode if self.sim_seed is not None else None
        seed = self.monitor.seed(seed)
        self.agent.seed(seed[0])  # Seed the agent with the main environment seed
        return seed

    def reset(self):
        self.observation = self.monitor.reset()
        self.agent.reset()

    def close(self):
        """
            Close the evaluation.
        """
        if self.training:
            self.save_agent_model("final")
        self.monitor.close()
        self.writer.close()
        if self.close_env:
            self.env.close()
