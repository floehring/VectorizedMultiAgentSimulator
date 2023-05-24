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
        n_agents = kwargs.get("n_agents", 5)
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
            collide=False,
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
        # The weights for the three rules
        alignment_weight = 1
        cohesion_weight = 1
        separation_weight = 1
        edge_avoidance_weight = 5

        action = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        action += self.steer_towards(agent, self.separation(agent, world)) * separation_weight
        action += self.steer_towards(agent, self.alignment(agent, world)) * alignment_weight
        action += self.steer_towards(agent, self.cohesion(agent, world)) * cohesion_weight
        action += self.steer_towards(agent, self.avoid_edges(agent, world)) * edge_avoidance_weight

        print(f'Agent: {agent.name} [vel: {agent.state.vel}, action: {agent.action.u}]')

        agent.action.u = action.clamp(-agent.u_range, agent.u_range)

    def separation(self, agent, world):
        perception_range = 0.3
        separation_distance = 0.1
        neighbors = []
        separation_heading = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        for boid in world.agents:
            if boid == agent:
                continue

            offset = boid.state.pos - agent.state.pos
            distance = torch.linalg.vector_norm(offset)

            if distance < perception_range:
                neighbors.append(boid)
                if distance < separation_distance:
                    separation_heading -= offset / distance

        if len(neighbors) == 0:
            return separation_heading
        return separation_heading / len(neighbors)

    def alignment(self, agent, world):
        perception_range = 0.3
        neighbors = []
        alignment_heading = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        for boid in world.agents:
            if boid == agent:
                continue

            offset = boid.state.pos - agent.state.pos
            distance = torch.linalg.vector_norm(offset)

            if distance < perception_range:
                alignment_heading += boid.state.vel
                neighbors.append(boid)

        if len(neighbors) == 0:
            return alignment_heading
        return alignment_heading / len(neighbors)

    def cohesion(self, agent, world):
        perception_range = 0.3
        neighbors = []
        cohesion_heading = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        for boid in world.agents:
            if boid == agent:
                continue

            offset = boid.state.pos - agent.state.pos
            distance = torch.linalg.vector_norm(offset)

            if distance < perception_range:
                cohesion_heading += boid.state.pos
                neighbors.append(boid)

        if len(neighbors) == 0:
            return cohesion_heading
        return cohesion_heading / len(neighbors)

    def steer_towards(self, agent, heading):
        print(f'Heading: {heading}')
        if not torch.all(torch.eq(heading, 0)):
            return heading - agent.state.vel
        return agent.state.vel

    def avoid_edges(self, agent, world):
        edge_margin = 0.1  # Margin to keep away from the edges
        avoidance_distance = 0.5  # Distance at which the avoidance force is applied

        desired_heading = torch.zeros((world.batch_dim, world.dim_p), device=world.device)

        # Calculate the distances from the agent's position to the edges of the world
        x_distance = torch.abs(agent.state.pos[:, 0]) - (world.x_semidim - edge_margin)
        y_distance = torch.abs(agent.state.pos[:, 1]) - (world.y_semidim - edge_margin)

        # Calculate the avoidance direction based on the distances
        x_avoidance_dir = torch.where(x_distance < 0, 0, -torch.sign(agent.state.pos[:, 0]))
        y_avoidance_dir = torch.where(y_distance < 0, 0, -torch.sign(agent.state.pos[:, 1]))

        # Combine the avoidance directions to form the desired heading
        desired_heading[:, 0] = x_avoidance_dir
        desired_heading[:, 1] = y_avoidance_dir

        # Normalize the desired heading to unit length
        desired_heading = torch.nn.functional.normalize(desired_heading, dim=-1)

        # Scale the desired heading by the desired speed
        desired_speed = agent.max_speed
        desired_heading *= desired_speed

        return desired_heading


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=False)
