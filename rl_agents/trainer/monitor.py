import json
import logging
import os
import csv
import time

from gym.wrappers import Monitor
from gym.wrappers.monitor import detect_training_manifests, collapse_env_infos, merge_stats_files
from gym.wrappers.monitoring import video_recorder
from gym.wrappers.monitoring.stats_recorder import StatsRecorder
from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np

logger = logging.getLogger(__name__)


class MonitorV2(Monitor):
    """
        A modified Environment Monitor that includes the following features:

        - The stats recorder is a StatsRecorderV2, that implements additional features, including the environment seed
        at each episode so that they can be reproduced
        - The wrapped environment is closed with the monitor
        - Video recording of all frames during an ongoing environment step
        - Automatic saving of all fields of the stats recorder
    """
    TRACKER_FILENAME = "experiment_tracker.csv"
    TRACKER_FIELDNAMES = ["scenario_id", "info"]

    def __init__(self, env, directory, video_callable=None, force=False, resume=True,
                 write_upon_reset=False, uid=None, mode=None, timing=False, options=None):
        # path-like objects only supported since python 3.6,
        # see https://python.readthedocs.io/en/stable/library/os.path.html#os.path.exists
        directory = str(directory)

        super(MonitorV2, self).__init__(env, directory, video_callable, force, resume, write_upon_reset, uid, mode)

        self.options = options
        self.timing = timing
        # Rodo
        self._save_experiment_info(directory, env)
        if self.options != None:
            self.video_callable = self.is_episode_selected_for_videosave

    def _save_experiment_info(self, directory, env):
        # Rodo
        # info_filename = "0_" + env.config["info"] + ".txt"
        # text_info_file = os.path.join(directory, info_filename)
        #
        # file  = open(text_info_file, "w+")
        # summary_msg = "info:" + env.config["info"] + '\n' + "other info"
        # file.write(summary_msg)
        # file.close()

        # add experiment info to tracker file
        if env.config["tracker_logging"]:
            tracker_path = os.path.join("..", "..", self.TRACKER_FILENAME)
            tracker_row = {"scenario_id": directory, "info": env.config["info"]}
            with open(tracker_path, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.TRACKER_FIELDNAMES)
                writer.writerow(tracker_row)

    def _video_enabled(self):
        return self.is_episode_selected_for_videosave(self.episode_id)
        # return self.video_callable(self.episode_id)

    def _start(self, directory, video_callable=None, force=False, resume=False, write_upon_reset=False, uid=None,
               mode=None):
        super(MonitorV2, self)._start(directory, video_callable, force, resume, write_upon_reset, uid, mode)
        self.stats_recorder = StatsRecorderV2(directory,
                                              '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix)
                                              , autoreset=self.env_semantics_autoreset, env_id=self.env.spec.id)

    def seed(self, seed=None):
        seeds = super(MonitorV2, self).seed(seed)
        self.stats_recorder.seed = seeds[0]  # Not sure why gym envs typically return *a list* of one seed
        return seeds

    def reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        # Rodo
        base_path = os.path.join(self.directory,
                                 '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),

        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(self.directory,
                                   '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled(),
        )

        # Instead of capturing just one frame, allow the environment to send all render frames when a step is ongoing
        if self._video_enabled() and hasattr(self.env.unwrapped, 'automatic_rendering_callback'):
            self.env.unwrapped.automatic_rendering_callback = self.video_recorder.capture_frame

    def _close_video_recorder(self):
        super(MonitorV2, self)._close_video_recorder()
        if hasattr(self.env.unwrapped, 'automatic_rendering_callback'):
            self.env.unwrapped.automatic_rendering_callback = None

    # Rodo
    def is_episode_selected(self):
        """
        :return: whether this episode was selected for rendering and model saving
        """
        return self._video_enabled()

    def is_episode_selected_for_modelsave(self):
        """
        :return: whether this episode was selected model saving
        """
        check_num = int(self.options['--model_save_freq'])
        return self.episode_id % check_num == 0 or self.options['--episodes'] == self.episode_id

    def is_episode_selected_for_videosave(self, episode_id):
        """
        :return: whether this episode was selected for rendering and model saving
        """
        check_num = int(self.options['--video_save_freq'])
        return self.episode_id % check_num == 0 or self.options['--episodes'] == (self.episode_id + 1)

    def step(self, action):
        if self.timing:
            before_step_start_time = time.time()
        self._before_step(action)
        if self.timing:
            before_step_elapsed_time = 1000 * (time.time() - before_step_start_time)
            print("INFO: ----before_step_elapsed_time Monitor + {:.1f}ms".format(before_step_elapsed_time))

        observation, reward, done, info = self.env.step(action)

        if self.timing:
            after_step_start_time = time.time()

        if isinstance(reward, tuple):  # Multi-agent setting Local state
            av = sum(reward) / len(reward)
            done = self._after_step(observation, sum(reward), done, info)
        else:
            done = self._after_step(observation, reward, done, info)

        if self.timing:
            after_step_elapsed_time = 1000 * (time.time() - after_step_start_time)
            print("INFO: ----after_elapsed_time Monitor + {:.1f}ms".format(after_step_elapsed_time))

        return observation, reward, done, info

    @property
    def monitor_id(self):
        return self._monitor_id

    @staticmethod
    def always_call_video(i):
        return True

    @staticmethod
    def load_results(training_dir):
        if not os.path.exists(training_dir):
            logger.error('Training directory %s not found', training_dir)
            return

        manifests = detect_training_manifests(training_dir)
        if not manifests:
            logger.error('No manifests found in training directory %s', training_dir)
            return

        logger.debug('Uploading data from manifest %s', ', '.join(manifests))

        # Load up stats + video files
        stats_files = []
        videos = []
        env_infos = []

        for manifest in manifests:
            with open(manifest) as f:
                contents = json.load(f)
                # Make these paths absolute again
                stats_files.append(os.path.join(training_dir, contents['stats']))
                videos += [(os.path.join(training_dir, v), os.path.join(training_dir, m))
                           for v, m in contents['videos']]
                env_infos.append(contents['env_info'])

        env_info = collapse_env_infos(env_infos, training_dir)

        # If several stats files are found, merge lists together and randomly pick single values
        all_contents = {}
        for file in stats_files:
            with open(file) as f:
                content = json.load(f)
                content.update(
                    {'manifests': manifests,
                     'env_info': env_info,
                     'videos': videos})
                if not all_contents:
                    all_contents.update(content)
                else:
                    for key, value in content.items():
                        if isinstance(value, list):
                            all_contents[key].extend(value)
                        else:
                            all_contents[key] = value
        return all_contents


class StatsRecorderV2(StatsRecorder):
    def __init__(self, directory, file_prefix, autoreset=False, env_id=None, log_infos=True):
        super(StatsRecorderV2, self).__init__(directory, file_prefix, autoreset, env_id)
        self.log_infos = log_infos

        # Rewards
        self.rewards_ = []
        self.episode_rewards_ = []

        # Infos
        self.infos = {}
        self.episode_infos = {}

        # Seed
        self.seed = None  # Set by the monitor when seeding the wrapped env
        self.episode_seeds = []

    def after_reset(self, observation):
        self.rewards_ = []
        self.infos = {}
        super(StatsRecorderV2, self).after_reset(observation)

    def after_step(self, observation, reward, done, info):
        # Aggregate rewards history
        self.rewards_.append(reward)
        if self.log_infos and info:
            for field, value in info.items():
                if field not in self.infos:
                    self.infos[field] = []
                try:  # Convert numpy types to serializable native python types
                    value = value.item()
                except AttributeError:
                    pass
                self.infos[field].append(value)

        super(StatsRecorderV2, self).after_step(observation, reward, done, info)

    def save_complete(self):
        if self.steps is not None:
            self.episode_rewards_.append(self.rewards_)
            for field, episode_values in self.infos.items():
                if field not in self.episode_infos:
                    self.episode_infos[field] = []
                self.episode_infos[field].append(episode_values)
            self.episode_seeds.append(self.seed)
            super(StatsRecorderV2, self).save_complete()

    def flush(self):
        if self.closed:
            return

        data = {
            'initial_reset_timestamp': self.initial_reset_timestamp,
            'timestamps': self.timestamps,
            'episode_lengths': self.episode_lengths,
            'episode_rewards': self.episode_rewards,
            'episode_rewards_': self.episode_rewards_,
            'episode_seeds': self.episode_seeds,
            'episode_types': self.episode_types,
        }
        for field, value in self.episode_infos.items():
            data["episode_{}".format(field)] = value

        # TODO: Behrad - I commented this because we are not using this JSON and it is a big file.
        # with atomic_write.atomic_write(self.path) as f:
        #     json.dump(data, f, default=json_encode_np)
