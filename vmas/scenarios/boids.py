import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):

    def __init__(self):
        super().__init__()

        # zoom out to see the whole world
        self.viewer_zoom = 1.5

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 8)
        n_obstacles = kwargs.get("n_obstacles", 0)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.15)

        self.plot_grid = True

        self.x_dim = 1
        self.y_dim = 1

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=5, drag=0, x_semidim=2, y_semidim=2)

        # Add agents
        self._target = Agent(
            name="target",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            max_speed=0.2
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
                max_speed=0.2
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
        ScenarioUtils.spawn_entities_randomly(
            self.world.entities,
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.x_dim, self.x_dim),
            y_bounds=(-self.y_dim, self.y_dim),
        )

        for agent in self.world.scripted_agents:
            agent.state.vel = 2 * torch.rand(self.world.batch_dim, self.world.dim_p, device=self.world.device) - 1
            # agent.state.vel = torch.tensor([[1.0, 0.0]], device=self.world.device)

    def reward(self, agent: Agent):
        # Reward not needed -> dummy values
        n_envs = agent.state.pos.shape[0]
        return torch.zeros(n_envs, device=self.world.device)

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ],
            dim=-1,
        )


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
        neighbors = [a for a in other_agents if torch.linalg.vector_norm(a.state.pos - agent.state.pos) <= perception_range]

        for neighbor in neighbors:

            # Rule 1: Alignment - steer towards the average heading of local flockmates
            alignment += neighbor.state.vel

        if len(neighbors) != 0:
            desired_velocity = alignment / len(neighbors)
            alignment_steering = desired_velocity - agent.state.vel

            agent.action.u = (alignment_weight * alignment_steering).clamp(-agent.u_range, agent.u_range)
        else:
            # If no neighbors, keep current velocity
            agent.action.u = torch.zeros((world.batch_dim, world.dim_p), device=world.device)

        print(f'Agent: {agent.name} [vel: {agent.state.vel}, action: {agent.action.u}]')


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=False)
