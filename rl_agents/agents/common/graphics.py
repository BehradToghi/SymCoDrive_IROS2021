from __future__ import division, print_function

from rl_agents.agents.deep_q_network.abstract import AbstractDQNAgent
from rl_agents.agents.deep_q_network.graphics import DQNGraphics


class AgentGraphics(object):
    """
        Graphical visualization of any Agent implementing AbstractAgent.
    """
    @classmethod
    def display(cls, agent, agent_surface, sim_surface=None):
        """
            Display an agent visualization on a pygame surface.

        :param agent: the agent to be displayed
        :param agent_surface: the pygame surface on which the agent is displayed
        :param sim_surface: the pygame surface on which the environment is displayed
        """

        if isinstance(agent, AbstractDQNAgent):
            DQNGraphics.display(agent, agent_surface, sim_surface)
