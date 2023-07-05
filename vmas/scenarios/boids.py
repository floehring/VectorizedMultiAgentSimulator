import torch
from pyglet.window import key

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y


class Scenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.obstacles = []
        self.obstacle_repulsion_zones = []
        # zoom out to see the whole world
        self.viewer_zoom = 1.8
        self.use_cohesion = True
        self.use_alignment = True
        self.use_separation = True

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 10)
        n_obstacles = kwargs.get("n_obstacles", 2)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.2)

        self.plot_grid = False
        self.margin = 1
        self.bounds = 6

        self.obstacle_radius = 0.3
        self.obstacle_repulsion_zone_radius = self.obstacle_radius + 0.4

        self.agent_radius = 0.1
        self.perception_range = self.agent_radius * 6
        self.obstacle_detection_range = self.perception_range * 1.1
        self.separation_distance = self.agent_radius * 3
        self.smoothing = 2

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=1, drag=0, x_semidim=self.bounds / 2,
                      y_semidim=self.bounds / 2)

        self.init_bounds(world)
        self.init_target(world)
        self.init_agents(world, n_agents, self.agent_radius, self.perception_range, self.separation_distance, self.obstacle_detection_range, self.smoothing)

        self.init_obstacles(world, n_obstacles)

        return world

    def init_agents(self, world, n_agents, agent_radius, perception_range, separation_distance,
                    obstacle_detection_range, smoothing):
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                render_action=True,
                action_script=BoidPolicy(self,
                                         obstacle_detection_range,
                                         perception_range,
                                         separation_distance,
                                         smoothing).run,
                max_speed=.20,
                shape=Sphere(radius=agent_radius),
                # sensors=[Lidar(world,
                #                max_range=obstacle_detection_range * 1.2,
                #                entity_filter=lambda e: e.name.startswith("obstacle"),
                #                render_color=Color.RED, n_rays=12)],
            )

            world.add_agent(agent)

    def init_target(self, world):
        self.target = Agent(
            name="target",
            collide=False,
            color=Color.GREEN,
            alpha=.3,
            render_action=True,
            max_speed=1,
            drag=0.1,
            shape=Sphere(radius=0.15),
        )
        world.add_agent(self.target)
        self.target_enabled = False

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
            self.add_landmark(world)

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            [agent for agent in self.world.scripted_agents],
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.world.x_semidim + self.margin, self.world.x_semidim - self.margin),
            y_bounds=(-self.world.y_semidim + self.margin, self.world.y_semidim - self.margin),
        )

        self.target.state.pos = torch.zeros(self.world.batch_dim, self.world.dim_p, device=self.world.device)

        ScenarioUtils.spawn_entities_randomly(
            [obstacle for obstacle in self.world.landmarks if obstacle.name.startswith("obstacle")],
            self.world,
            env_index,
            self._min_dist_between_entities * 4,
            x_bounds=(-self.world.x_semidim + self.margin, self.world.x_semidim - self.margin),
            y_bounds=(-self.world.y_semidim + self.margin, self.world.y_semidim - self.margin),
        )

        for repulsion_zone in self.obstacle_repulsion_zones:
            repulsion_zone.state.pos = repulsion_zone.parent.state.pos

        for agent in self.world.scripted_agents:
            agent.state.vel = torch.nn.functional.normalize(
                2 * torch.rand(self.world.batch_dim, self.world.dim_p, device=self.world.device) - 1) * agent.max_speed

    def reward(self, agent: Agent):
        # Reward not needed -> dummy values
        n_envs = agent.state.pos.shape[0]
        return torch.zeros(n_envs, device=self.world.device)

    def observation(self, agent: Agent):
        return torch.cat([torch.zeros_like(agent.state.pos), torch.zeros_like(agent.state.vel)], dim=-1)

    def handle_key_press(self, env, k):
        if k == key.SPACE:
            self.target_enabled = not self.target_enabled
            self.target._alpha = 1 if self.target_enabled else .3
        elif k == key.P:
            self.add_landmark(self.world, self.target.state.pos)
        elif k == key.B:
            self.toggle_behavior("cohesion")
        elif k == key.N:
            self.toggle_behavior("alignment")
        elif k == key.M:
            self.toggle_behavior("separation")
        elif k == key.H:
            self.add_agent(self.world, self.target.state.pos)


    def handle_key_release(self, env, key: int):
        pass

    def add_agent(self, world, pos=None):
        agent = Agent(
            name=f"agent_{len(world.scripted_agents)}",
            collide=False,
            render_action=True,
            action_script=BoidPolicy(self,
                                     self.obstacle_detection_range,
                                     self.perception_range,
                                     self.separation_distance,
                                     self.smoothing).run,
            max_speed=.20,
            shape=Sphere(radius=self.agent_radius),
            # sensors=[Lidar(world,
            #                max_range=self.obstacle_detection_range * 1.2,
            #                entity_filter=lambda e: e.name.startswith("obstacle"),
            #                render_color=Color.RED, n_rays=12)],
        )
        world.add_agent(agent)
        agent.state.pos = pos if pos is not None else torch.zeros(world.batch_dim, world.dim_p, device=world.device)

    def toggle_behavior(self, behavior):
        if behavior == "cohesion":
            self.use_cohesion = not self.use_cohesion
        elif behavior == "alignment":
            self.use_alignment = not self.use_alignment
        elif behavior == "separation":
            self.use_separation = not self.use_separation
        print(f"Use cohesion: {self.use_cohesion}, alignment: {self.use_alignment}, separation: {self.use_separation}")

    def add_landmark(self, world, pos=None):
        obstacle = Landmark(
            name=f"obstacle_{len(self.obstacles)}",
            collide=True,
            movable=False,
            shape=Sphere(radius=self.obstacle_radius),
            color=Color.RED,
        )
        world.add_landmark(obstacle)
        if pos is not None:
            obstacle.state.pos = pos
        self.obstacles.append(obstacle)


class BoidPolicy:

    def __init__(self, scenario, obstacle_detection_range, perception_range, separation_distance, smoothing):
        self.scenario = scenario
        self.perception_range = perception_range
        self.separation_distance = separation_distance
        self.smoothing = smoothing
        self.obstacle_detection_range = obstacle_detection_range

    def run(self, agent, world):
        # The weights for the three rules
        alignment_weight = 1
        cohesion_weight = 0.5
        separation_weight = 3
        obstacle_avoidance_weight = 3
        boundary_avoidance_weight = 0.5

        alignment = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        cohesion = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        separation = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        action = torch.zeros((world.batch_dim, world.dim_p), device=world.device)

        # How strong is the attraction to the target
        target_factor = 6
        target = self.scenario.target

        neighbor_count = 0
        for boid in world.agents:
            if boid == agent or (not self.scenario.target_enabled and boid == target):
                continue

            offset = agent.state.pos - boid.state.pos
            distance = torch.linalg.norm(offset)

            if self.is_in_perception_range(boid, distance, offset, self.perception_range):
                alignment += boid.state.vel
                cohesion += boid.state.pos

                if distance < self.separation_distance:
                    separation += offset / distance
                neighbor_count += 1

        if neighbor_count > 0:
            alignment /= neighbor_count
            cohesion /= neighbor_count
            cohesion -= agent.state.pos

            if self.scenario.use_separation:
                action += ((agent.state.vel + separation) * separation_weight) * (1 / self.smoothing)

            if self.scenario.use_alignment:
                action += ((agent.state.vel + alignment) * alignment_weight) * (1 / self.smoothing)

            if self.scenario.use_cohesion:
                action += ((agent.state.vel + cohesion) * cohesion_weight) * (1 / self.smoothing)

        avoid_boundaries_action = self.avoid_boundaries(agent, world)
        action += avoid_boundaries_action * boundary_avoidance_weight

        avoid_obstacles_action = self.avoid_obstacles(agent, world)
        action += avoid_obstacles_action * obstacle_avoidance_weight

        if self.scenario.target_enabled:
            action += ((target.state.pos - agent.state.pos) - agent.state.vel) * target_factor

        epsilon = 1e-5
        agent.action.u = action.clamp(-agent.u_range + epsilon, agent.u_range - epsilon)

    def is_in_perception_range(self, boid, distance, offset, perception_range):
        fov = 320

        boid_direction = torch.nn.functional.normalize(boid.state.vel)
        relative_position_of_other = torch.nn.functional.normalize(offset)

        cos_angle = torch.dot(boid_direction.squeeze(), relative_position_of_other.squeeze())
        angle = torch.acos(cos_angle)

        return distance < perception_range and angle < fov / 2

    def get_heading(self, agent, heading):
        return agent.state.vel + heading * (1 / self.smoothing)

    def avoid_boundaries(self, agent, world):
        # Define a margin for edge avoidance. Adjust this value based on your world size.
        margin = self.scenario.margin
        max_boundary_push = agent.max_speed * 8

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
        max_avoidance = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        Lb = self.perception_range * 1.3
        maxspeed = agent.max_speed * 1.5

        obstacles = self.scenario.obstacles

        for obstacle in obstacles:
            dx = obstacle.state.pos - agent.state.pos
            dy = agent.state.vel
            rb = torch.linalg.norm(dy)
            ro = self.scenario.obstacle_radius

            if torch.linalg.norm(dx) <= Lb:
                if torch.linalg.norm(dy) <= rb + ro:
                    n = -1 * dx / torch.linalg.norm(dx)
                    e = ((rb + ro) - torch.linalg.norm(dy)) / (rb + ro)
                    e *= maxspeed
                    steer = e * n

                    # if this obstacle requires a stronger avoidance action, update max_avoidance
                    if torch.linalg.norm(steer) > torch.linalg.norm(max_avoidance):
                        max_avoidance = steer

        return max_avoidance

    def avoid_obstacles_lidar(self, agent, world):
        obstacle_avoidance = torch.zeros((world.batch_dim, world.dim_p), device=world.device)

        lidar = agent.sensors[0]
        measurements = lidar.measure()

        agent_direction = torch.atan2(agent.state.vel[:, 1], agent.state.vel[:, 0])

        fov = torch.pi / 2

        min_angle = float('inf')
        min_angle_dir = None

        dy = agent.state.vel
        rb = torch.linalg.norm(dy)
        ro = self.scenario.obstacle_radius

        for i, dist in enumerate(measurements[0]):

            ray_angle = lidar._angles[i]
            relative_angle = torch.abs(ray_angle - agent_direction)
            relative_angle = torch.min(relative_angle, 2 * torch.pi - relative_angle)

            if dist < self.obstacle_detection_range and relative_angle < fov / 2:
                if relative_angle < min_angle:
                    min_angle = relative_angle
                    min_angle_dir = self.get_ray_direction(ray_angle)

        if min_angle_dir is not None:
            avoidance_strength = ((rb + ro) - torch.linalg.norm(dy)) / (rb + ro)
            avoidance_direction = -min_angle_dir
            avoidance_direction /= torch.linalg.norm(avoidance_direction)
            obstacle_avoidance += avoidance_strength * avoidance_direction

        return obstacle_avoidance


    @staticmethod
    def linear_interpolation(margin, dist, max):
        return (margin - dist) * max / margin

    @staticmethod
    def get_ray_direction(angle):
        ray_dir_world = torch.stack(
            [torch.cos(angle), torch.sin(angle)], dim=-1
        )

        return ray_dir_world


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=False, display_info=False)
