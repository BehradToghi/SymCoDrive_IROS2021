from abc import ABC, abstractmethod
import numpy as np
from gym import spaces
import sys
import time
import time
from rl_agents.agents.common.abstract import AbstractStochasticAgent
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.memory import ReplayMemory, Transition


class AbstractDQNAgent(AbstractStochasticAgent, ABC):
    def __init__(self, env, config=None):
        super(AbstractDQNAgent, self).__init__(config)
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete) or isinstance(env.action_space, spaces.Tuple), \
            "Only compatible with Discrete action spaces."
        self.memory = ReplayMemory(self.config)
        self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space,env = self.env)
        self.training = True
        self.previous_state = None

    @classmethod
    def default_config(cls):
        return dict(model=dict(type="DuelingNetwork"),
                    optimizer=dict(type="ADAM",
                                   lr=5e-4,
                                   weight_decay=0,
                                   k=5),
                    loss_function="l2",
                    memory_capacity=50000,
                    batch_size=100,
                    gamma=0.99,
                    device="cuda:best",
                    exploration=dict(method="EpsilonGreedy"),
                    target_update=1,
                    double=True)

    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition by performing a Deep Q-Network iteration

            - push the transition into memory
            - sample a minibatch
            - compute the bellman residual loss over the minibatch
            - perform one gradient descent step
            - slowly track the policy network with the target network
        :param state: a state
        :param action: an action
        :param reward: a reward
        :param next_state: a next state
        :param done: whether state is terminal
        """
        # start = time.time()
        replay_memory_info = None
        if not self.training:
            return
        # TODO
        # Global reward need to be with global state or will cause confusion to network
        # if isinstance(state, tuple) and isinstance(action, tuple):  # Multi-agent setting
        #     [self.memory.push(agent_state, agent_action, reward, agent_next_state, done, info)
        #      for agent_state, agent_action, agent_next_state in zip(state, action, next_state)]
        if isinstance(state, tuple) and isinstance(action, tuple):  # Multi-agent setting

            [self.memory.push(agent_state, agent_action, reward, agent_next_state, done, info)
             for agent_state, agent_action, reward, agent_next_state in zip(state, action, reward, next_state)]
            # time1 = time.time()
            # debug
            # for agent_state, agent_action, reward, agent_next_state in zip(state, action, reward, next_state):
            #     print(agent_state, agent_action, reward, agent_next_state)
        else:  # Single-agent setting
            self.memory.push(state, action, reward, next_state, done, info)

        replay_memory_info = [0, 0]
        if self.config["calculate_replay_size"]:
            replay_memory_info = [len(self.memory.memory),
                                  self.get_size(self.memory.memory) / 1000000]  # length and RAM usage
        # time2 = time.time()
        batch = self.sample_minibatch()
        time3 = time.time()
        if batch:
            # print("batch={:.3f}MB, memory={:.3f}MB len={:n}".format(self.get_size(batch) / 1000000, self.get_size(self.memory.memory) / 1000000, len(self.memory.memory)))
            loss, _, _ = self.compute_bellman_residual(batch)
            # time4 = time.time()
            self.step_optimizer(loss)
            # time5 = time.time()
            self.update_target_network()
            # time6 = time.time()

            # print(">>>>>> memory push: {:.3f}ms, "
            #       "update info: {:.3f}ms, "
            #       "sample batch: {:.3f}ms, "
            #       "bellman loss: {:.3f}ms, "
            #       "optimizer: {:.3f}ms, "
            #       "update target: {:.3f}ms".format(
            #     (time1 - start) * 1000,
            #     (time2 - time1) * 1000,
            #     (time3 - time2) * 1000,
            #     (time4 - time3) * 1000,
            #     (time5 - time4) * 1000,
            #     (time6 - time5) * 1000))

        return replay_memory_info

    def get_size(self, obj, seen=None):
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([self.get_size(v, seen) for v in obj.values()])
            size += sum([self.get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += self.get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([self.get_size(i, seen) for i in obj])
        return size

    def act(self, state, step_exploration_time=True):
        """
            Act according to the state-action value model and an exploration policy
        :param state: current state
        :param step_exploration_time: step the exploration schedule
        :return: an action
        """
        self.previous_state = state
        if step_exploration_time:
            self.exploration_policy.step_time()
        # Handle multi-agent observations
        # TODO: it would be more efficient to forward a batch of states
        if isinstance(state, tuple):
            return tuple(self.act(agent_state, step_exploration_time=False) for agent_state in state)

        # Single-agent setting
        timing = False
        if timing:
            predict_start_time = time.time()

        values = self.get_state_action_values(state)

        if timing:
            predict_elapsed_time = 1000 * (time.time() - predict_start_time)
            print("INFO: --Action Foward Prediction time 1 Agent + {:.1f}ms".format(predict_elapsed_time))
        self.exploration_policy.update(values)
        return self.exploration_policy.sample()

    def sample_minibatch(self):
        if len(self.memory) < self.config["batch_size"] * self.config["batch_counter_wait"]:
            return None
        transitions = self.memory.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))

    def update_target_network(self):
        self.steps += 1
        if self.steps % self.config["target_update"] == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())

    @abstractmethod
    def compute_bellman_residual(self, batch, target_state_action_value=None):
        """
            Compute the Bellman Residual Loss over a batch
        :param batch: batch of transitions
        :param target_state_action_value: if provided, acts as a target (s,a)-value
                                          if not, it will be computed from batch and model (Double DQN target)
        :return: the loss over the batch, and the computed target
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_state_values(self, states):
        """
        Get the state values of several states
        :param states: [s1; ...; sN] an array of states
        :return: values, actions:
                 - [V1; ...; VN] the array of the state values for each state
                 - [a1*; ...; aN*] the array of corresponding optimal action indexes for each state
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_state_action_values(self, states):
        """
        Get the state-action values of several states
        :param states: [s1; ...; sN] an array of states
        :return: values:[[Q11, ..., Q1n]; ...] the array of all action values for each state
        """
        raise NotImplementedError

    def get_state_value(self, state):
        """
        :param state: s, an environment state
        :return: V, its state-value
        """
        values, actions = self.get_batch_state_values([state])
        return values[0], actions[0]

    def get_state_action_values(self, state):
        """
        :param state: s, an environment state
        :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions
        """
        return self.get_batch_state_action_values([state])[0]

    def step_optimizer(self, loss):
        raise NotImplementedError

    def seed(self, seed=None):
        return self.exploration_policy.seed(seed)

    def reset(self):
        pass

    def set_writer(self, writer):
        super().set_writer(writer)
        try:
            self.exploration_policy.set_writer(writer)
        except AttributeError:
            pass

    def action_distribution(self, state):
        self.previous_state = state
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values)
        return self.exploration_policy.get_distribution()

    def set_time(self, time):
        self.exploration_policy.set_time(time)

    def eval(self):
        self.training = False
        self.config['exploration']['method'] = "Greedy"
        self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space)
