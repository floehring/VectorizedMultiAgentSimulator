import torch
from pyglet.window import key

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, X, Y


class Scenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.obstacles = []
        # zoom out to see the whole world
        self.viewer_zoom = 2.2

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 1)
        n_obstacles = kwargs.get("n_obstacles", 0)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.15)

        self.plot_grid = True

        self.x_dim = 1
        self.y_dim = 1
        self.margin = 1

        self.obstacle_radius = 0.3
        self.obstacle_repulsion_zone_radius = self.obstacle_radius + 0.3

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=1, drag=0, x_semidim=4, y_semidim=4)

        # Add target
        self.target = Agent(
            name="target",
            collide=False,
            color=Color.GREEN,
            alpha=1,
            render_action=True,
            max_speed=1,
            drag=0.1,
            shape=Sphere(radius=0.15),
            # action_script=self.action_script_creator(),
        )
        # world.add_agent(self.target)
        self.target_enabled = False

        # Add agents
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                render_action=True,
                action_script=BoidPolicy(self).run,
                max_speed=0.15,
                shape=Sphere(radius=0.08),
            )

            world.add_agent(agent)

        # Add obstacle
        obstacle = Landmark(
            name=f"obstacle",
            collide=True,
            movable=False,
            shape=Sphere(radius=self.obstacle_radius),
            color=Color.RED,
        )

        world.add_landmark(obstacle)
        self.obstacles.append(obstacle)

        # Add obstacle repulsion zone
        obstacle_repulsion_zone = Agent(
            name=f"obstacle_repulsion_zone",
            collide=False,
            movable=False,
            shape=Sphere(radius=self.obstacle_repulsion_zone_radius),
            color=Color.BLUE,
        )
        obstacle_repulsion_zone._alpha = 0.1
        world.add_agent(obstacle_repulsion_zone)

        # self.init_bounds(world)
        # self.init_obstacles(world, n_obstacles)

        return world

    def init_bounds(self, world):
        self.bounds = []
        boundary_box = world.add_landmark(
            Landmark(
                name="boundary_box",
                collide=False,
                movable=False,
                shape=Box(length=world.y_semidim * 2, width=world.x_semidim * 2, hollow=True),
                color=Color.GRAY,
            )
        )
        self.bounds.append(boundary_box)

        margin_box = world.add_landmark(
            Landmark(
                name="margin_box",
                collide=False,
                movable=False,
                shape=Box(length=world.y_semidim * 2 - self.margin, width=world.x_semidim * 2 - self.margin,
                          hollow=True),
                color=Color.WHITE,
            )
        )
        self.bounds.append(margin_box)

    def init_obstacles(self, world, n_obstacles):
        # Add landmarks
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.4),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

    def reset_world_at(self, env_index: int = None):
        # ScenarioUtils.spawn_entities_randomly(
        #     [entity for entity in self.world.entities if
        #      entity.name != 'margin_box' and entity.name != 'boundary_box' and entity != self.target],
        #     self.world,
        #     env_index,
        #     self._min_dist_between_entities,
        #     x_bounds=(-self.x_dim, self.x_dim),
        #     y_bounds=(-self.y_dim, self.y_dim),
        # )

        self.world.scripted_agents[0].state.pos = torch.tensor([[-self.world.x_semidim, 0.0]], device=self.world.device)
        self.world.landmarks[0].state.pos = torch.tensor([[0.0, 0.0]], device=self.world.device)
        obstacle_repulsion_zone = [agent for agent in self.world.policy_agents if agent.name == 'obstacle_repulsion_zone'][0]
        obstacle_repulsion_zone.state.pos = torch.tensor([[0.0, 0.0]], device=self.world.device)


        # for agent in self.world.scripted_agents:
        #     agent.state.vel = 2 * torch.rand(self.world.batch_dim, self.world.dim_p, device=self.world.device) - 1

    def reward(self, agent: Agent):
        # Reward not needed -> dummy values
        n_envs = agent.state.pos.shape[0]
        return torch.zeros(n_envs, device=self.world.device)

    def observation(self, agent: Agent):
        return torch.cat([torch.zeros_like(agent.state.pos), torch.zeros_like(agent.state.vel)], dim=-1)

    def handle_key_press(self, env, k):
        if k == key.SPACE:
            print("Toggle target")
            self.target_enabled = not self.target_enabled
            self.target._alpha = 1 if self.target_enabled else .3
        elif k == key.P:
            obstacle = Landmark(
                name=f"obstacle_{len(self.obstacles)}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.4),
                color=Color.RED,
            )
            self.world.add_landmark(obstacle)
            obstacle.state.pos = self.target.state.pos
            self.obstacles.append(obstacle)

    def handle_key_release(self, env, key: int):
        pass


class BoidPolicy:

    def __init__(self, scenario):
        self.scenario = scenario

    def run(self, agent, world):
        # The weights for the three rules
        alignment_weight = 2
        cohesion_weight = 1
        separation_weight = 1.5
        avoidance_weight = 4

        separation_distance = agent.shape.circumscribed_radius() * 6

        perception_range = 0.6

        alignment = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        cohesion = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        separation = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        action = torch.zeros((world.batch_dim, world.dim_p), device=world.device)

        # How strong is the attraction to the target
        target_factor = 2
        target = self.scenario.target

        neighbor_count = 0
        for boid in world.agents:
            if boid == agent or (not self.scenario.target_enabled and boid == target):
                continue

            offset = boid.state.pos - agent.state.pos
            distance = torch.linalg.norm(offset)

            if distance < perception_range:
                alignment += boid.state.vel
                cohesion += boid.state.pos

                if distance < separation_distance:
                    separation -= offset / distance
                neighbor_count += 1

        if neighbor_count > 0:
            alignment /= neighbor_count
            cohesion /= neighbor_count

            print(f'Alignment: {alignment}, Cohesion: {cohesion}, Separation: {separation}')

            action += self.steer_towards(agent, separation) * separation_weight
            action += self.steer_towards(agent, alignment) * alignment_weight
            action += self.steer_towards(agent, cohesion) * cohesion_weight

        else:

            action += torch.zeros((world.batch_dim, world.dim_p), device=world.device)

        action += self.avoid_boundaries(agent, world)
        action += self.steer_towards(agent, self.avoid_obstacles(agent, world)) * avoidance_weight

        if self.scenario.target_enabled:
            action += self.steer_towards(agent, target.state.pos - agent.state.pos) * target_factor

        epsilon = 1e-5

        # Let agent move from left to right
        obstacle_action = self.avoid_obstacles(agent, world)
        agent.action.u = torch.tensor([[1.0, 0]], device=world.device) + obstacle_action

        # agent.action.u = action.clamp(-agent.u_range + epsilon, agent.u_range - epsilon)

        # print(f'Agent: {agent.name} [vel: {agent.state.vel}, action: {agent.action.u}, pos: {agent.state.pos}, mag: {torch.linalg.norm(agent.state.vel)}] \n')

    def steer_towards(self, agent, heading):
        """Steer towards a heading vector, as described by Craig Reynolds' Boids algorithm.
        
            steering = desired_velocity - current_velocity
        """
        if not torch.all(torch.eq(heading, 0)):
            return torch.nn.functional.normalize(heading) * agent.max_speed - agent.state.vel
        return agent.state.vel

    def avoid_boundaries(self, agent, world):
        # Define a margin for edge avoidance. Adjust this value based on your world size.
        margin = self.scenario.margin
        max_push = agent.max_speed * 4

        # Create an avoidance heading initially as zeros.
        avoidance_heading = torch.zeros_like(agent.state.vel)

        # Check proximity to each of the four edges and adjust heading away from the edge if too close
        for axis in [X, Y]:
            dist_rt = world.x_semidim - agent.state.pos[:, axis]  # dist for right and top
            dist_lb = torch.abs(-world.x_semidim - agent.state.pos[:, axis])  # dist for left and bottom
            if dist_rt < margin:
                avoidance_heading[:, axis] = -torch.sign(agent.state.pos[:, axis]) \
                                             * self.linear_interpolation(margin, dist_rt, max_push)
            if dist_lb < margin:
                avoidance_heading[:, axis] = -torch.sign(agent.state.pos[:, axis]) \
                                             * self.linear_interpolation(margin, dist_lb, max_push)

        # print(f'Avoidance heading: {avoidance_heading} \n')

        return avoidance_heading

    def avoid_obstacles(self, agent, world):
        max_push = agent.max_speed * 4
        avoidance_heading = torch.zeros_like(agent.state.vel)

        # Check proximity to each obstacle and adjust heading away from the obstacle if too close
        for obstacle in self.scenario.obstacles:
            dist_to_obstacle = torch.linalg.norm(agent.state.pos - obstacle.state.pos)
            if dist_to_obstacle < obstacle_detection_radius:
                avoidance_heading += (agent.state.pos - obstacle.state.pos) * \
                                     self.linear_interpolation(obstacle_detection_radius, dist_to_obstacle, max_push)

        print(f'Obstacle avoidance heading: {avoidance_heading} \n')

        return avoidance_heading

    def linear_interpolation(self, margin, dist, max):
        return (margin - dist) * max / margin


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=False)
