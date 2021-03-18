"""
Usage:
  experiments evaluate [options]
  experiments -h | --help

Options:
  -h --help                    Show this screen.
  --episodes <count>           Number of episodes [default: 10].
  --no-display                 Disable environment, agent, and rewards rendering [default: False].
  --name-from-config           Name the output folder from the corresponding config files
  --processes <count>          Number of running processes [default: 4].
  --recover                    Load model from the latest checkpoint [default: False].
  --recover-from <file>        Load model from a given checkpoint.
  --seed <str>                 Seed the environments and agents.
  --train                      Train the agent
  --test                       Test the agent
  --episodes_test <count>      Number of episodes for test if test is set [default: 10].
  --verbose                    Set log level to debug instead of info.
  --repeat <times>             Repeat several times [default: 1].
  --model_save_freq <count>    Save a model every n episodes [default: 500].
  --video_save_freq <count>    Save a video every n episodes [default: 500].
  --create_episode_log         Create episodes logs.
  --individual_episode_log_level <count>   Individual logs 0: no individual log,2: individual log for controlled only,3: individual log for all vehicles [default: 2].
  --create_timestep_log        Create timesteps logs.
  --timestep_log_freq <count>  Save a timestep log every n episodes [default: 1].
  --environment <str>          Environment config [default: configs/default/env.json].
  --agent <str>                Agent config [default: None].
"""
import datetime
import os
from pathlib import Path
import gym
import json
from docopt import docopt
from itertools import product
from multiprocessing.pool import Pool
from collections import Counter
import linecache
import tracemalloc

from rl_agents.trainer import logger
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

# os.environ['SDL_VIDEODRIVER'] = 'dummy'

BENCHMARK_FILE = 'benchmark_summary'
LOGGING_CONFIG = 'configs/logging.json'
VERBOSE_CONFIG = 'configs/verbose.json'


def main() -> None:
    opts = docopt(__doc__)
    if opts['evaluate']:
        for _ in range(int(opts['--repeat'])):
            evaluate(environment_config=opts['--environment'], agent_config=opts['--agent'], options=opts)
    elif opts['benchmark']:
        benchmark(opts)


def evaluate(environment_config="configs/default/env.json", agent_config=None, options=None):
    """
        Evaluate an agent interacting with an environment.

    :param environment_config: the path of the environment configuration file
    :param agent_config: the path of the agent configuration file
    :param options: the evaluation options
    """

    logger.configure(LOGGING_CONFIG)

    if options['--verbose']:
        logger.configure(VERBOSE_CONFIG)
    run_directory = None

    if options['--name-from-config']:
        run_directory = "{}_{}_{}".format(Path(agent_config).with_suffix('').name,
                                          datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                                          os.getpid())
    options['--seed'] = int(options['--seed']) if options['--seed'] is not None else None

    # TODO
    #
    # options['--no-display']=False if (options['--no-display'])=='False' else True
    # options['--recover'] = False if (options['--recover'])=='False' else True
    # Rodo
    # env_info(env)

    # TODO
    #
    # if options['--train']:
    #     options['--no-display']=True
    #     environment_config['offscreen_rendering']= True
    # else:
    #     options['--no-display'] = False
    #     environment_config['offscreen_rendering'] = False
    #     options['--recover']="./out/HighwayEnv/DQNAgent/saved_models/latest.tar"

    env = load_environment(environment_config)
    if agent_config == "None":
        agent_config = env.config["agent_config"]
        if "auto_tau" in agent_config["exploration"] and (agent_config["exploration"]["auto_tau"]):
            agent_config["exploration"]["tau"] = env.config["policy_frequency"] * env.config["duration"] * int(options['--episodes'] * env.config["controlled_vehicles"] )/ 50
    agent = load_agent(agent_config, env)
    # TODO diferent display options for agent, env, rewards
    evaluation_train = Evaluation(env,
                            agent,
                            run_directory=run_directory,
                            num_episodes=int(options['--episodes']),
                            sim_seed=options['--seed'],
                            recover=options['--recover'] or options['--recover-from'],
                            display_env=not options['--no-display'],
                            display_agent=not options['--no-display'],
                            display_rewards=not options['--no-display'],
                            training=options['--train'],
                            create_episode_log=options['--create_episode_log'],
                            individual_episode_log_level=int(options['--individual_episode_log_level']),
                            create_timestep_log=options['--create_timestep_log'],
                            timestep_log_freq=options['--timestep_log_freq'],
                            options=options)

    if options['--train']:
        evaluation_train.train()
    else:
        evaluation_train.close()
    if options['--test']:
        agent_test = load_agent(agent_config, env)
        if options['--train']:
            agent_test = evaluation_train.agent

        evaluation_test = Evaluation(env,
                                      agent_test,
                                      run_directory=run_directory,
                                      num_episodes=int(options['--episodes_test']),
                                      sim_seed=options['--seed'],
                                      recover=options['--recover'] or options['--recover-from'],
                                      display_env=not options['--no-display'],
                                      display_agent=not options['--no-display'],
                                      display_rewards=not options['--no-display'],
                                      training=False,
                                      test=options['--test'],
                                      create_episode_log=options['--create_episode_log'],
                                      individual_episode_log_level=int(options['--individual_episode_log_level']),
                                      create_timestep_log=options['--create_timestep_log'],
                                      timestep_log_freq=options['--timestep_log_freq'],
                                      options=options)

        evaluation_test.test()


    return os.path.relpath(evaluation_train.monitor.directory)


def benchmark(options):
    """
        Run the evaluations of several agents interacting in several environments.

    The evaluations are dispatched over several processes.
    The benchmark configuration file should look like this:
    {
        "environments": ["path/to/env1.json", ...],
        "agents: ["path/to/agent1.json", ...]
    }

    :param options: the evaluation options, containing the path to the benchmark configuration file.
    """
    # Prepare experiments
    with open(options['<benchmark>']) as f:
        benchmark_config = json.loads(f.read())
    generate_agent_configs(benchmark_config)
    experiments = product(benchmark_config['environments'], benchmark_config['agents'], [options])

    # Run evaluations
    with Pool(processes=int(options['--processes'])) as pool:
        results = pool.starmap(evaluate, experiments)

    # Clean temporary config files
    generate_agent_configs(benchmark_config, clean=True)

    # Write evaluations summary
    benchmark_filename = os.path.join(Evaluation.OUTPUT_FOLDER, '{}_{}.{}.json'.format(
        BENCHMARK_FILE, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid()))
    with open(benchmark_filename, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
        gym.logger.info('Benchmark done. Summary written in: {}'.format(benchmark_filename))


def generate_agent_configs(benchmark_config, clean=False):
    """
        Generate several agent configurations from:
        - a "base_agent" configuration path field
        - a "key" field referring to a parameter that should vary
        - a "values" field listing the values of the parameter taken for each agent

        Created agent configurations will be stored in temporary file, that can be removed after use by setting the
        argument clean=True.
    :param benchmark_config: a benchmark configuration
    :param clean: should the temporary agent configurations files be removed
    :return the updated benchmark config
    """
    if "base_agent" in benchmark_config:
        with open(benchmark_config["base_agent"], 'r') as f:
            base_config = json.load(f)
            configs = [dict(base_config, **{benchmark_config["key"]: value})
                       for value in benchmark_config["values"]]
            paths = [
                Path(benchmark_config["base_agent"]).parent / "bench_{}={}.json".format(benchmark_config["key"], value)
                for value in benchmark_config["values"]]
            if clean:
                [path.unlink() for path in paths]
            else:
                [json.dump(config, path.open('w')) for config, path in zip(configs, paths)]
            benchmark_config["agents"] = paths
    return benchmark_config

def malloc_display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))



if __name__ == "__main__":
    # tracemalloc.start()

    main()

    # snapshot = tracemalloc.take_snapshot()
    # malloc_display_top(snapshot)
