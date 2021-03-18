import logging
import torch
from gym import spaces
import numpy as np
import time

from rl_agents.agents.common.memory import Transition
from rl_agents.agents.common.models import model_factory, size_model_config, trainable_parameters
from rl_agents.agents.common.optimizers import loss_function_factory, optimizer_factory
from rl_agents.agents.common.utils import choose_device
from rl_agents.agents.deep_q_network.abstract import AbstractDQNAgent
from torchsummary import summary
logger = logging.getLogger(__name__)


class DQNAgent(AbstractDQNAgent):
    def __init__(self, env, config=None):
        super(DQNAgent, self).__init__(env, config)
        size_model_config(self.env, self.config["model"])  # in = obs space , out= action space
        self.value_net = model_factory(self.config["model"])
        self.target_net = model_factory(self.config["model"])
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        logger.debug("Number of trainable parameters: {}".format(trainable_parameters(self.value_net)))
        self.device = choose_device(self.config["device"])
        self.value_net.to(self.device)
        self.target_net.to(self.device)
        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self.value_net.parameters(),
                                           **self.config["optimizer"])
        self.steps = 0

        # inputc = (self.config["model"]["in_channels"], self.config["model"]["in_height"] ,  self.config["model"]["in_width"])
        # # # summary(self.value_net , input , batch_size=-1)
        model = self.value_net.to(self.device)
        inputc = self.config["model"]["inputc"]
        summary(model,  input_size = inputc)
        # repr(self.value_net)
        print(self.value_net)

    def step_optimizer(self, loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # start = time.time()
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(tuple(torch.tensor(np.array([batch.state]), dtype=torch.float))).to(self.device)
            # print ("point1 diff = {:.3f} ms".format((time.time()-start)*1000))
            action = torch.tensor(np.array(batch.action), dtype=torch.long).to(self.device)
            # print("point2 diff = {:.3f} ms".format((time.time() - start) * 1000))
            reward = torch.tensor(np.array(batch.reward), dtype=torch.float).to(self.device)
            # print("point3 diff = {:.3f} ms".format((time.time() - start) * 1000))
            next_state = torch.cat(tuple(torch.tensor(np.array([batch.next_state]), dtype=torch.float))).to(self.device)
            # print("point4 diff = {:.3f} ms".format((time.time() - start) * 1000))
            # #TODO :test wihtout converting to int
            # if self.env.config["observation"]["observation_config"]["type"] == "HeatmapObservation":
            #     scale = 255.0
            #     state = torch.div(state, scale)
            #     # print("point4.1 diff = {:.3f} ms".format((time.time() - start) * 1000))
            #     next_state = torch.div(next_state, scale)

            # print("point5 diff = {:.3f} ms".format((time.time() - start) * 1000))
            terminal = torch.tensor(np.array(batch.terminal), dtype=torch.bool).to(self.device)
            # print("point6 diff = {:.3f} ms".format((time.time() - start) * 1000))
            batch = Transition(state, action, reward, next_state, terminal, batch.info)
            # print("point7 diff = {:.3f} ms".format((time.time() - start) * 1000))


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.value_net(batch.state)
        # print("point8 diff = {:.3f} ms".format((time.time() - start) * 1000))
        state_action_values = state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)
        # print("point9 diff = {:.3f} ms".format((time.time() - start) * 1000))

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net(batch.next_state).max(1)
                    # Double Q-learning: estimate action values from target network
                    best_values = self.target_net(batch.next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    best_values, _ = self.target_net(batch.next_state).max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = batch.reward + self.config["gamma"] * next_state_values

        # Compute loss
        # print("point10 diff = {:.3f} ms".format((time.time() - start) * 1000))
        loss = self.loss_function(state_action_values, target_state_action_value)
        # print("point11 diff = {:.3f} ms".format((time.time() - start) * 1000))
        return loss, target_state_action_value, batch

    def get_batch_state_values(self, states):
        start = time.time()
        # values, actions = self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).max(1)
        # print("point1 diff = {:.3f} ms".format((time.time() - start) * 1000))
        values, actions = self.value_net(torch.tensor(np.array(states), dtype=torch.float).to(self.device)).max(1)
        # print("point2 diff = {:.3f} ms".format((time.time() - start) * 1000))
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        start = time.time()
        # val = self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).data.cpu().numpy()
        # print("point1 diff = {:.3f} ms".format((time.time() - start) * 1000))
        val = self.value_net(torch.tensor(np.array(states), dtype=torch.float).to(self.device)).data.cpu().numpy()
        # print("point2 diff = {:.3f} ms".format((time.time() - start) * 1000))
        return val

    def save(self, filename):
        state = {'state_dict': self.value_net.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def set_writer(self, writer):
        super().set_writer(writer)
        obs_shape = self.env.observation_space.shape if isinstance(self.env.observation_space, spaces.Box) else \
            self.env.observation_space.spaces[0].shape
        model_input = torch.zeros((1, *obs_shape), dtype=torch.float, device=self.device)
        self.writer.add_graph(self.value_net, input_to_model=(model_input,)),
        self.writer.add_scalar("agent/trainable_parameters", trainable_parameters(self.value_net), 0)
