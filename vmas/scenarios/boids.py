import torch
from pyglet.window import key

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

smoothing = 1


class Scenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.obstacles = []
        self.obstacle_repulsion_zones = []
        # zoom out to see the whole world
        self.viewer_zoom = 2.2

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 5)
        n_obstacles = kwargs.get("n_obstacles", 0)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.15)

        self.plot_grid = False

        self.x_dim = 1
        self.y_dim = 1
        self.margin = 1

        self.obstacle_radius = 0.3
        self.obstacle_repulsion_zone_radius = self.obstacle_radius + 0.4

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=1,drag=0, x_semidim=4, y_semidim=4)

        self.init_bounds(world)

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
        world.add_agent(self.target)
        self.target_enabled = False

        # Add agents
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                render_action=True,
                action_script=BoidPolicy(self).run,
                max_speed=0.15,
                shape=Sphere(radius=0.1),
            )

            world.add_agent(agent)

        # # Add obstacle
        # obstacle = Landmark(
        #     name=f"obstacle",
        #     collide=True,
        #     movable=False,
        #     shape=Sphere(radius=self.obstacle_radius),
        #     color=Color.RED,
        # )
        #
        # world.add_landmark(obstacle)
        # self.obstacles.append(obstacle)
        #
        # # Add obstacle repulsion zone
        # obstacle_repulsion_zone = Agent(
        #     name=f"obstacle_repulsion_zone",
        #     collide=False,
        #     movable=False,
        #     shape=Sphere(radius=self.obstacle_repulsion_zone_radius),
        #     color=Color.BLUE,
        # )
        # obstacle_repulsion_zone._alpha = 0.1
        # world.add_agent(obstacle_repulsion_zone)


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
        ScenarioUtils.spawn_entities_randomly(
            [agent for agent in self.world.scripted_agents],
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.x_dim, self.x_dim),
            y_bounds=(-self.y_dim, self.y_dim),
        )

        for agent in self.world.scripted_agents:
            agent.state.vel = 2 * torch.rand(self.world.batch_dim, self.world.dim_p, device=self.world.device) - 1
        print("Reset world")

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
            self.add_landmark()

    def handle_key_release(self, env, key: int):
        pass


    def add_landmark(self):
        obstacle = Landmark(
            name=f"obstacle_{len(self.obstacles)}",
            collide=True,
            movable=False,
            shape=Sphere(radius=self.obstacle_radius),
            color=Color.RED,
        )
        self.world.add_landmark(obstacle)
        obstacle.state.pos = self.target.state.pos
        self.obstacles.append(obstacle)

        obstacle_repulsion_zone = Agent(
            name=f"obstacle_repulsion_zone{len(self.obstacle_repulsion_zones)}",
            collide=False,
            movable=False,
            shape=Sphere(radius=self.obstacle_repulsion_zone_radius),
            color=Color.BLUE,
        )
        obstacle_repulsion_zone._alpha = 0.1
        self.world.add_agent(obstacle_repulsion_zone)
        obstacle_repulsion_zone.state.pos = self.target.state.pos
        self.obstacle_repulsion_zones.append(obstacle_repulsion_zone)
class BoidPolicy:

    def __init__(self, scenario, perception_range=0.8, separation_distance=0.2):
        self.scenario = scenario
        # self.perception_range = perception_range
        self.separation_distance = separation_distance


    def run(self, agent, world):
        # The weights for the three rules
        alignment_weight = 3
        cohesion_weight = 1
        separation_weight = 5
        avoidance_weight = 4

        perception_range = agent.shape.circumscribed_radius() * 6
        separation_distance = perception_range / 2

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

            if self.is_in_perception_range(boid, agent, distance, offset, perception_range):
                alignment += boid.state.vel
                cohesion += boid.state.pos
                print(f'Boid pos: {boid.state.pos}')

                if distance < separation_distance:
                    # separation -= offset / distance
                    separation -= offset
                neighbor_count += 1


        if neighbor_count > 0:
            alignment /= neighbor_count
            cohesion /= neighbor_count
            cohesion -= agent.state.pos


            action += self.steer_towards(agent, separation) * separation_weight
            action += self.steer_towards(agent, alignment) * alignment_weight
            action += self.steer_towards(agent, cohesion) * cohesion_weight

        print('\n')
        avoid_boundaries_action = self.avoid_boundaries(agent, world)
        print('Avoid boundaries action: ', avoid_boundaries_action)
        action += avoid_boundaries_action

        avoid_obstacles_action = self.avoid_obstacles(agent, world)
        print('Avoid obstacles action: ', avoid_obstacles_action)
        action += avoid_obstacles_action * avoidance_weight

        if self.scenario.target_enabled:
            action += self.steer_towards(agent, target.state.pos - agent.state.pos) * target_factor

        epsilon = 1e-5

        print('Action: ', action)
        agent.action.u = action.clamp(-agent.u_range + epsilon, agent.u_range - epsilon)

    def is_in_perception_range(self, boid, other, distance, offset, perception_range):
        fov = 300

        boid_direction = torch.nn.functional.normalize(boid.state.vel)
        relative_position_of_other = torch.nn.functional.normalize(offset)

        cos_angle = torch.dot(boid_direction.squeeze(), relative_position_of_other.squeeze())
        angle = torch.acos(cos_angle)

        return distance < perception_range and angle < fov / 2

    @staticmethod
    def steer_towards(agent, heading):
        """Steer towards a heading vector, as described by Craig Reynolds' Boids algorithm.
        
            steering = desired_velocity - current_velocity
        """
        # if not torch.all(torch.eq(heading, 0)):
        #     return torch.nn.functional.normalize(heading) * agent.max_speed - agent.state.vel
        # return agent.state.vel
        return agent.state.vel + heading * (1 / smoothing)

    def avoid_boundaries(self, agent, world):
        # Define a margin for edge avoidance. Adjust this value based on your world size.
        margin = self.scenario.margin
        max_boundary_push = agent.max_speed * 10

        # Create an avoidance heading initially as zeros.
        avoidance_heading = torch.zeros_like(agent.state.vel)

        # Check proximity to each of the four edges and adjust heading away from the edge if too close
        for axis in [X, Y]:
            dist_rt = world.x_semidim - agent.state.pos[:, axis]  # dist for right and top
            dist_lb = torch.abs(-world.x_semidim - agent.state.pos[:, axis])  # dist for left and bottom
            if dist_rt < margin:
                avoidance_heading[:, axis] = -torch.sign(agent.state.pos[:, axis]) \
                                             * self.linear_interpolation(margin, dist_rt, max_boundary_push)
            if dist_lb < margin:
                avoidance_heading[:, axis] = -torch.sign(agent.state.pos[:, axis]) \
                                             * self.linear_interpolation(margin, dist_lb, max_boundary_push)

        return avoidance_heading

    def avoid_obstacles(self, agent, world):
        max_obstacle_push = agent.max_speed * 4
        avoidance_heading = torch.zeros_like(agent.state.vel)

        # Check proximity to each obstacle and adjust heading away from the obstacle if too close
        for obstacle in self.scenario.obstacles:
            dir_to_boid = agent.state.pos - obstacle.state.pos
            dist_to_boid = torch.linalg.norm(dir_to_boid)

            if dist_to_boid < self.scenario.obstacle_repulsion_zone_radius:
                dir_to_boid /= dist_to_boid
                repulsion_strength = torch.abs(self.linear_interpolation(self.scenario.obstacle_repulsion_zone_radius, dist_to_boid, max_obstacle_push))
                print("repulsion_strength: ", repulsion_strength)
                avoidance_heading += dir_to_boid * repulsion_strength

        return avoidance_heading

    @staticmethod
    def linear_interpolation(margin, dist, max):
        return (margin - dist) * max / margin


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=False, display_info=False)
