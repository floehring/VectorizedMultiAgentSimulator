from typing import Callable, Dict

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, X, Y, ScenarioUtils


class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        n_obstacles = kwargs.get("n_obstacles", 5)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.15)

        self.collision_reward = kwargs.get("collision_reward", -0.1)
        self.dist_shaping_factor = kwargs.get("dist_shaping_factor", 1)

        self.plot_grid = True
        self.desired_distance = 0.1
        self.min_collision_distance = 0.005

        self.x_dim = 1
        self.y_dim = 1

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=5)

        # Add agents
        self._target = Agent(
            name="target",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            # action_script=self.action_script_creator(),
        )
        world.add_agent(self._target)

        # Add agents
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                render_action=True,
                action_script=BoidPolicy().run,
            )

            world.add_agent(agent)

        # Add landmarks
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.1),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        return world

    def reset_world_at(self, env_index: int = None):

        self.t = torch.zeros(self.world.batch_dim, device=self.world.device)

        ScenarioUtils.spawn_entities_randomly(
            self.world.entities,
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.x_dim, self.x_dim),
            y_bounds=(-self.y_dim, self.y_dim),
        )

    def reward(self, agent: Agent):
        # Reward not needed -> dummy values
        n_envs = agent.state.pos.shape[0]
        return torch.zeros(n_envs, device=self.world.device)

    def observation(self, agent: Agent):
        print(f'observation {agent.name}')
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ],
            dim=-1,
        )

    def action_script_creator(self):
        def action_script(agent, world):
            t = self.t / 30
            agent.action.u = torch.stack([torch.cos(t), torch.sin(t)], dim=1)

        return action_script


class BoidPolicy:

    def run(self, agent, world):
        # The perception range within which the agent considers other agents
        perception_range = 0.5

        # The weights for the three rules
        alignment_weight = 1.0
        cohesion_weight = 1.0
        separation_weight = 1.5

        # Initialize vectors for the three rules
        alignment = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        cohesion = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        separation = torch.zeros((world.batch_dim, world.dim_p), device=world.device)

        other_agents = [a for a in world.agents if a != agent]
        neighbors = [a for a in other_agents if torch.linalg.vector_norm(a.state.pos - agent.state.pos) < perception_range]

        for neighbor in neighbors:

            # Rule 1: Alignment - steer towards the average heading of local flockmates
            alignment += neighbor.state.vel

        # Average the vectors
        alignment /= len(neighbors)

        # No neighbors -> agent maintains its current velocity
        if len(neighbors) == 0:
            agent.action.u = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        else:
            agent.action.u = alignment_weight * alignment

        print(f'Agent: {agent.name}; Physical Action: {agent.action.u}')


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=False)
