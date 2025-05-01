"""
Jan 20
We add multiple ants to the env
"""

import tempfile, mujoco
import xml.etree.ElementTree as ET

import numpy as np
from gymnasium.spaces import Box

from ogbench.locomaze.ant import AntEnv
from ogbench.locomaze.humanoid import HumanoidEnv
from ogbench.locomaze.point import PointEnv
from ogbench.luo_utils.d4rl_m2d_const import get_str_maze_spec
import ogbench.utils as utils

from diffuser.utils.ogb_paper_vis_utils import ogb_vis_create_multi_ant_environment
## TODO: From Here Jan 20 19:52, 
# 1. Finish the multi Ant Visualization Pipeline Tonigit
# 2. Write something in the related work section as well.
# 3. Wrap up Exp and some appendix.

def ogb_make_maze_env_Multi_Agent_Luo(loco_env_type, maze_env_type, num_agents, *args, **kwargs):
    """Factory function for creating a maze environment.

    Args:
        loco_env_type: Locomotion environment type. One of 'point', 'ant', or 'humanoid'.
        maze_env_type: Maze environment type. Either 'maze' or 'ball'.
        *args: Additional arguments to pass to the target class.
        **kwargs: Additional keyword arguments to pass to the target class.
    """
    if loco_env_type == 'point':
        loco_env_class = PointEnv
    elif loco_env_type == 'ant':
        loco_env_class = AntEnv
    elif loco_env_type == 'humanoid':
        loco_env_class = HumanoidEnv
    else:
        raise ValueError(f'Unknown locomotion environment type: {loco_env_type}')

    print(f'{loco_env_class=}', flush=True)

    class MazeEnv_Multi_Agent_Luo(loco_env_class):
        """Maze environment.

        It inherits from the locomotion environment and adds a maze to it.
        """

        def __init__(
            self,
            maze_type='large',
            maze_unit=4.0,
            maze_height=0.5,
            terminate_at_goal=True,
            ob_type='states',
            add_noise_to_goal=True,
            seed_addn=None, ## aka, seed_add_noise
            luo_cfg={},
            *args,
            **kwargs,
        ):
            """Initialize the maze environment.

            Args:
                maze_type: Maze type. One of 'arena', 'medium', 'large', 'giant', or 'teleport'.
                maze_unit: Size of a maze unit block.
                maze_height: Height of the maze walls.
                terminate_at_goal: Whether to terminate the episode when the goal is reached.
                ob_type: Observation type. Either 'states' or 'pixels'.
                add_noise_to_goal: Whether to add noise to the goal position.
                seed_addn: if not None, use this int as rng seed; ogb default is no seed
                *args: Additional arguments to pass to the parent locomotion environment.
                **kwargs: Additional keyword arguments to pass to the parent locomotion environment.
            """
            self._maze_type = maze_type
            self._maze_unit = maze_unit
            self._maze_height = maze_height
            self._terminate_at_goal = terminate_at_goal
            self._ob_type = ob_type
            self._add_noise_to_goal = add_noise_to_goal
            assert ob_type in ['states', 'pixels']

            # Define constants.
            self._offset_x = 4
            self._offset_y = 4
            self._noise = 1
            self._goal_tol = 1.0 if loco_env_type == 'point' else 0.5

            ## ------ Added By Luo ------
            self.seed_addn = seed_addn
            if seed_addn is not None:
                self.rng_addn = np.random.default_rng(seed=seed_addn)

            ## --------------------------
            self.is_ball_freejnt = luo_cfg.get('is_ball_freejnt', True) ## just for vis
            

            # Define maze map.
            self._teleport_info = None
            if self._maze_type == 'arena':
                maze_map = [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'medium':
                maze_map = [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'large':
                maze_map = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'giant':
                maze_map = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                    [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'teleport':
                maze_map = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                    [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
                    [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
                self._teleport_info = dict(
                    teleport_in_ijs=[(4, 6), (5, 1)],
                    teleport_out_ijs=[(1, 7), (6, 1), (6, 10)],
                    teleport_radius=1,
                )
                self._teleport_info['teleport_in_xys'] = [
                    self.ij_to_xy(ij) for ij in self._teleport_info['teleport_in_ijs']
                ]
                self._teleport_info['teleport_out_xys'] = [
                    self.ij_to_xy(ij) for ij in self._teleport_info['teleport_out_ijs']
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.maze_map = np.array(maze_map)

            # Update XML file.
            xml_file = self.xml_file
            tree = ET.parse(xml_file)

            ## TODO: Jan 20
            tree = ogb_vis_create_multi_ant_environment(tree, num_agents=num_agents)
            ## TODO:

            self.update_tree(tree)
            

            _, maze_xml_file = tempfile.mkstemp(text=True, suffix='.xml')
            tree.write(maze_xml_file)

            super().__init__(xml_file=maze_xml_file, *args, **kwargs)

            # Set task goals.
            self.task_infos = []
            self.cur_task_id = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)
            self.cur_goal_xy = np.zeros(2)

            if self._ob_type == 'pixels':
                self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

                # Manually color the floor to enable the agent to infer its position from the observation.
                tex_grid = self.model.tex('grid')
                tex_height = tex_grid.height[0]
                tex_width = tex_grid.width[0]
                # MuJoCo 3.2.1 changed the attribute name from 'tex_rgb' to 'tex_data'.
                attr_name = 'tex_rgb' if hasattr(self.model, 'tex_rgb') else 'tex_data'
                tex_rgb = getattr(self.model, attr_name)[tex_grid.adr[0] : tex_grid.adr[0] + 3 * tex_height * tex_width]
                tex_rgb = tex_rgb.reshape(tex_height, tex_width, 3)
                for x in range(tex_height):
                    for y in range(tex_width):
                        min_value = 0
                        max_value = 192
                        r = int(x / tex_height * (max_value - min_value) + min_value)
                        g = int(y / tex_width * (max_value - min_value) + min_value)
                        tex_rgb[x, y, :] = [r, g, 128]
            else:
                ex_ob = self.get_ob()
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

            # Set camera.
            self.reset()
            self.render()
            self.mujoco_renderer.viewer.cam.lookat[0] = 2 * (self.maze_map.shape[1] - 3)
            self.mujoco_renderer.viewer.cam.lookat[1] = 2 * (self.maze_map.shape[0] - 3)
            self.mujoco_renderer.viewer.cam.distance = 5 * (self.maze_map.shape[1] - 2)
            self.mujoco_renderer.viewer.cam.elevation = -90

            ## ------ Added By Luo ------
            self.str_maze_spec = get_str_maze_spec(maze_type)
            
            ## --------------------------
            

        def update_tree(self, tree):
            """Update the XML tree to include the maze."""
            worldbody = tree.find('.//worldbody')

            # Add walls.
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    struct = self.maze_map[i, j]
                    if struct == 1:
                        ET.SubElement(
                            worldbody,
                            'geom',
                            name=f'block_{i}_{j}',
                            pos=f'{j * self._maze_unit - self._offset_x} {i * self._maze_unit - self._offset_y} {self._maze_height / 2 * self._maze_unit}',
                            size=f'{self._maze_unit / 2} {self._maze_unit / 2} {self._maze_height / 2 * self._maze_unit}',
                            type='box',
                            contype='1',
                            conaffinity='1',
                            material='wall',
                        )

            # Adjust floor size.
            center_x, center_y = 2 * (self.maze_map.shape[1] - 3), 2 * (self.maze_map.shape[0] - 3)
            size_x, size_y = 2 * self.maze_map.shape[1], 2 * self.maze_map.shape[0]
            floor = tree.find('.//geom[@name="floor"]')
            floor.set('pos', f'{center_x} {center_y} 0')
            floor.set('size', f'{size_x} {size_y} 0.2')

            if self._teleport_info is not None:
                # Add teleports.
                for i, (x, y) in enumerate(self._teleport_info['teleport_in_xys']):
                    ET.SubElement(
                        worldbody,
                        'geom',
                        name=f'teleport_in_{i}',
                        type='cylinder',
                        size=f'{self._teleport_info["teleport_radius"]} .05',
                        pos=f'{x} {y} .05',
                        material='teleport_in',
                        contype='0',
                        conaffinity='0',
                    )
                for i, (x, y) in enumerate(self._teleport_info['teleport_out_xys']):
                    ET.SubElement(
                        worldbody,
                        'geom',
                        name=f'teleport_out_{i}',
                        type='cylinder',
                        size=f'{self._teleport_info["teleport_radius"]} .05',
                        pos=f'{x} {y} .05',
                        material='teleport_out',
                        contype='0',
                        conaffinity='0',
                    )

            if self._ob_type == 'pixels':
                # Color wall.
                wall = tree.find('.//material[@name="wall"]')
                wall.set('rgba', '.6 .6 .6 1')
                # Remove ambient light.
                light = tree.find('.//light[@name="global"]')
                light.attrib.pop('ambient')
                # Remove torso light.
                torso_light = tree.find('.//light[@name="torso_light"]')
                torso_light_parent = tree.find('.//light[@name="torso_light"]/..')
                torso_light_parent.remove(torso_light)
                # Remove texture repeat.
                grid = tree.find('.//material[@name="grid"]')
                grid.set('texuniform', 'false')
                if loco_env_type == 'ant':
                    # Color one leg white to break symmetry.
                    tree.find('.//geom[@name="aux_1_geom"]').set('material', 'self_white')
                    tree.find('.//geom[@name="left_leg_geom"]').set('material', 'self_white')
                    tree.find('.//geom[@name="left_ankle_geom"]').set('material', 'self_white')
            else:
                # Only show the target for states-based observation.
                ET.SubElement(
                    worldbody,
                    'geom',
                    name='target',
                    type='cylinder',
                    size='1.0 .05',
                    pos='0 0 .05',
                    material='target',
                    contype='0',
                    conaffinity='0',
                    density='0',
                )
                ## Jan 2, Add one geom to indicate the current goal waypoint 
                ## check if neg pos will affect simulation
                ET.SubElement(
                    worldbody,
                    'geom',
                    name='subgoal_waypnt', ## subgoal waypoint
                    type='cylinder',
                    size='1.0 .05',
                    pos='-10 0 .05',
                    material='subgoal_waypnt',
                    contype='0',
                    conaffinity='0',
                    density='0',
                )
                ## NEW Feb 2, for better vis
                ET.SubElement(
                    worldbody,
                    'geom',
                    name='start_marker', ## subgoal waypoint
                    type='cylinder',
                    size='1.0 .05',
                    pos='-10 0 .05',
                    rgba='0.08 0.4 0.75 0.7',
                    contype='0',
                    conaffinity='0',
                    density='0',
                )

        def set_tasks(self):
            # `tasks` is a list of tasks, where each task is a list of two tuples: (init_ij, goal_ij).
            if self._maze_type == 'arena':
                tasks = [
                    [(1, 1), (6, 6)],
                ]
            elif self._maze_type == 'medium':
                tasks = [
                    [(1, 1), (6, 6)],
                    [(6, 1), (1, 6)],
                    [(5, 3), (4, 2)],
                    [(6, 5), (6, 1)],
                    [(2, 6), (1, 1)],
                ]
            elif self._maze_type == 'large':
                tasks = [
                    [(1, 1), (7, 10)],
                    [(5, 4), (7, 1)],
                    [(7, 4), (1, 10)],
                    [(3, 8), (5, 4)],
                    [(1, 1), (5, 4)],
                ]
            elif self._maze_type == 'giant':
                tasks = [
                    [(1, 1), (10, 14)],
                    [(1, 14), (10, 1)],
                    [(8, 14), (1, 1)],
                    [(8, 3), (5, 12)],
                    [(5, 9), (3, 8)],
                ]
            elif self._maze_type == 'teleport':
                tasks = [
                    [(1, 10), (7, 1)],
                    [(1, 1), (7, 10)],
                    [(5, 6), (7, 10)],
                    [(7, 1), (7, 10)],
                    [(5, 6), (7, 1)],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.task_infos = []
            for i, task in enumerate(tasks):
                self.task_infos.append(
                    dict(
                        task_name=f'task{i + 1}',
                        init_ij=task[0],
                        init_xy=self.ij_to_xy(task[0]),
                        goal_ij=task[1],
                        goal_xy=self.ij_to_xy(task[1]),
                    )
                )

        def reset(self, options=None, *args, **kwargs):
            if options is None:
                options = {}
            # Set the task goal.
            if 'task_id' in options:
                # Use the pre-defined task.
                assert 1 <= options['task_id'] <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = options['task_id']
                ## task_infos: a list of dict
                # {'task_name': 'task1',
                # 'init_ij': (1, 1),
                # 'init_xy': (0.0, 0.0),
                # 'goal_ij': (10, 14),
                # 'goal_xy': (52.0, 36.0)},
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_info' in options:
                # Use the provided task information.
                self.cur_task_id = None
                self.cur_task_info = options['task_info']
            else:
                # Randomly sample a task.
                self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]

            # Whether to provide a rendering of the goal.
            render_goal = False
            if 'render_goal' in options:
                render_goal = options['render_goal']

            # Get initial and goal positions with noise.
            init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['init_ij']))
            goal_xy = self.ij_to_xy(self.cur_task_info['goal_ij'])
            if self._add_noise_to_goal:
                goal_xy = self.add_noise(goal_xy)

            ## Control 3
            ## sample a seed for resetting the env (for goal state)
            if self.seed_addn is not None: ## and 'seed' not in kwargs
                if 'seed' in kwargs:
                    utils.print_color(f'[OgB MazeEnv] Seed Conflict, Ignore Ori: {kwargs=}', c='y')
                kwargs['seed'] = self.sample_an_int()
            # First, force set the position to the goal position to obtain the goal observation.
            super().reset(*args, **kwargs)

            ## --- Dec 27 added by luo, make start/goal reproducible ---
            ## Three Seeds of interest when reseting:
            ## 1. seed for adding noise to the xy-level start and goal
            ## 2. seed when doing action_space sampling
            ## 3. seed when reset the agent, will affect qpos/qvel.
            ## 'rng_addn' can control all of them
            
            ## Control 2
            utils.print_color(f'[OgB MazeEnv] call reset (for goal) {args=} {kwargs=}')
            if self.seed_addn is not None:
                self.action_space.seed( self.sample_an_int() )
            ## ---------------------------------------------------------

            # Do a few random steps to stabilize the environment.
            num_random_actions = 40 if loco_env_type == 'humanoid' else 5
            for _ in range(num_random_actions):
                super().step(self.action_space.sample())

            # Save the goal observation.
            self.set_goal(goal_xy=goal_xy)
            self.set_xy(goal_xy)
            goal_ob = self.get_ob()
            goal_full_jnt_state = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

            if render_goal:
                goal_rendered = self.render()

            ## Control 3
            ## sample a seed for resetting the env (for start state)
            if self.seed_addn is not None:
                kwargs['seed'] = self.sample_an_int()

            # Now, do the actual reset.
            ob, info = super().reset(*args, **kwargs)
            self.set_goal(goal_xy=goal_xy)
            self.set_xy(init_xy)
            ob = self.get_ob()
            info['goal'] = goal_ob
            info['goal_full_jnt_state'] = goal_full_jnt_state
            if render_goal:
                info['goal_rendered'] = goal_rendered

            return ob, info

        def step(self, action):
            ob, reward, terminated, truncated, info = super().step(action)

            if self._teleport_info is not None:
                # Check if the agent is close to a inbound teleport.
                for x, y in self._teleport_info['teleport_in_xys']:
                    if np.linalg.norm(self.get_xy() - np.array([x, y])) <= self._teleport_info['teleport_radius'] * 1.5:
                        # Teleport the agent to a random outbound teleport.
                        teleport_out_xy = self._teleport_info['teleport_out_xys'][
                            np.random.randint(len(self._teleport_info['teleport_out_xys']))
                        ]
                        self.set_xy(np.array(teleport_out_xy))
                        break

            # Check if the agent has reached the goal.
            if np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= self._goal_tol:
                if self._terminate_at_goal:
                    terminated = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

            return ob, reward, terminated, truncated, info

        def get_ob(self, ob_type=None):
            ob_type = self._ob_type if ob_type is None else ob_type
            if ob_type == 'states':
                return super().get_ob()
            else:
                frame = self.render()
                return frame

        def set_goal(self, goal_ij=None, goal_xy=None):
            """Set the goal position and update the target object."""
            if goal_xy is None:
                self.cur_goal_xy = self.ij_to_xy(goal_ij)
                if self._add_noise_to_goal:
                    self.cur_goal_xy = self.add_noise(self.cur_goal_xy)
            else:
                self.cur_goal_xy = goal_xy
            if self._ob_type == 'states':
                self.model.geom('target').pos[:2] = goal_xy

        def get_oracle_subgoal(self, start_xy, goal_xy):
            """Get the oracle subgoal for the agent.

            If the goal is unreachable, it returns the current position as the subgoal.

            Args:
                start_xy: Starting position of the agent.
                goal_xy: Goal position of the agent.
            Returns:
                A tuple of the oracle subgoal and the BFS map.
            """
            start_ij = self.xy_to_ij(start_xy)
            goal_ij = self.xy_to_ij(goal_xy)

            # Run BFS to find the next subgoal.
            bfs_map = self.maze_map.copy()
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    bfs_map[i][j] = -1

            bfs_map[goal_ij[0], goal_ij[1]] = 0
            queue = [goal_ij]
            while len(queue) > 0:
                i, j = queue.pop(0)
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < self.maze_map.shape[0]
                        and 0 <= nj < self.maze_map.shape[1]
                        and self.maze_map[ni, nj] == 0
                        and bfs_map[ni, nj] == -1
                    ):
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))

            # Find the subgoal that attains the minimum BFS value.
            subgoal_ij = start_ij
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = start_ij[0] + di, start_ij[1] + dj
                if (
                    0 <= ni < self.maze_map.shape[0]
                    and 0 <= nj < self.maze_map.shape[1]
                    and self.maze_map[ni, nj] == 0
                    and bfs_map[ni, nj] < bfs_map[subgoal_ij[0], subgoal_ij[1]]
                ):
                    subgoal_ij = (ni, nj)
            subgoal_xy = self.ij_to_xy(subgoal_ij)
            return np.array(subgoal_xy), bfs_map

        def xy_to_ij(self, xy):
            maze_unit = self._maze_unit
            i = int((xy[1] + self._offset_y + 0.5 * maze_unit) / maze_unit)
            j = int((xy[0] + self._offset_x + 0.5 * maze_unit) / maze_unit)
            return i, j

        def ij_to_xy(self, ij):
            i, j = ij
            x = j * self._maze_unit - self._offset_x
            y = i * self._maze_unit - self._offset_y
            return x, y


        # ### ------ Ori Implmentation ------
        # def add_noise(self, xy):
        #     random_x = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
        #     random_y = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
        #     return xy[0] + random_x, xy[1] + random_y
        # ## --------------------------------

        def set_seed_addn(self, seed_addn):
            self.seed_addn = seed_addn
            self.rng_addn = np.random.default_rng(seed=self.seed_addn)

        def sample_an_int(self):
            '''sample a int for reset env or action_space'''
            return self.rng_addn.integers(low=0, high=1000000).item() ## [0,1M]
        

        def add_noise(self, xy):
            if self.seed_addn is None:
                ## ogb default, without any seed
                random_x = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
                random_y = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            else:
                random_x = self.rng_addn.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
                random_y = self.rng_addn.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4

            return xy[0] + random_x, xy[1] + random_y



        def __del__(self):
            print(f'__del__ MazeEnv: {self}', flush=True)
            # self.close()
            # del self.mujoco_renderer
            # super().__del__() ## TODO: Dec 27 why no this func ??
        
        def get_offset_x(self):
            return self._offset_x

        def get_offset_y(self):
            return self._offset_y
        
        def get_maze_unit(self):
            return self._maze_unit

        def render(self, *args, **kwargs):
            ## Dec 31, added by Luo
            img_format = kwargs.get('img_format', 'cell_ij')
            if 'img_format' in kwargs:
                del kwargs['img_format']

            rd_cfg = {k: v for k,v in kwargs.get('rd_cfg', {}).items()}
            if 'rd_cfg' in kwargs:
                del kwargs['rd_cfg']
            print(f'{rd_cfg=}')


            img = super().render(*args, **kwargs)
            ## we want it in cell ij format
            if img_format == 'cell_ij':
                img_r = np.rot90(img,)
                img_r = np.fliplr(img_r,)
            elif img_format == 'ori':
                img_r = img

            is_crop_black = rd_cfg.get('is_crop_black', True)
            ## TODO:
            if is_crop_black:
                if self._maze_type == 'giant':
                    # w_factor = 0.08
                    # w_factor = 0.10
                    w_factor = 0.115 ## 800 resol
                    w_factor = 0.118 ## 2000 resol
                    ## fixed the multi agent diff cam pos issue:
                    w_factor = 0.08 ## 0.07 also looks good
                    
                elif self._maze_type == 'large':
                    # w_factor = 0.081
                    # w_factor = 0.118
                    ## change back to the below value, have fixed the multi-agent camera pose issue 
                    ## (previously the camera change pose in every init)
                    w_factor = 0.081
                elif self._maze_type == 'medium':
                    w_factor = 0.00 ## at Feb 10, update the init pos of multi ant
                    # w_factor = 0.11 ##
                    rd_cfg['h_factor'] = w_factor
                elif self._maze_type == 'arena':
                    w_factor = 0.00 ## Feb 20
                    rd_cfg['h_factor'] = w_factor
                else:
                    raise NotImplementedError
                
                ## NOTE:
                if rd_cfg.get('w_factor', None) is not None:
                    w_factor = rd_cfg['w_factor']
                    # rd_cfg['h_factor'] = w_factor

                print(f'crop: {w_factor=}')

                tmp_h, tmp_w = img_r.shape[0:2]
                tmp_st = int(tmp_w * w_factor)
                tmp_end = tmp_w - tmp_st
                img_r = img_r[:, tmp_st:tmp_end, :]

                if rd_cfg.get('h_factor', None):
                    h_factor = rd_cfg['h_factor']
                    tmp_st = int(tmp_h * h_factor)
                    tmp_end = tmp_h - tmp_st
                    img_r = img_r[tmp_st:tmp_end, :, :]
                    print(f'crop: {h_factor=}')

            
            return img_r
        
        def set_subgoal_waypnt(self, waypnt_xy):
            ## visualize the current subgoal
            if self._ob_type == 'states':
                self.model.geom('subgoal_waypnt').pos[:2] = waypnt_xy
                # if self.model.geom('subgoal_waypnt').size[0] == 0.0:
                    # self.model.geom('subgoal_waypnt').size[0] = 0.6


        def set_start_marker(self, start_xy):
            ## visualize the start position
            if self._ob_type == 'states':
                self.model.geom('start_marker').pos[:2] = start_xy


        def show_img(self, width=300, **kwargs):
            from mediapy import show_image
            img = self.render(**kwargs)
            show_image(img, width=width)
            # return img



    

    class BallEnv_Multi_Agent_Luo(MazeEnv_Multi_Agent_Luo):
        def update_tree(self, tree):
            super().update_tree(tree)

            # Add ball.
            worldbody = tree.find('.//worldbody')

            # ball = ET.SubElement(worldbody, 'body', name='ball', pos='0 0 0.5')
            # ET.SubElement(ball, 'freejoint', name='ball_root')
            # ET.SubElement(
            #     ball,
            #     'geom',
            #     name='ball',
            #     size='.25',
            #     material='ball',
            #     priority='1',
            #     conaffinity='1',
            #     condim='6',
            # )
            # ET.SubElement(ball, 'light', name='ball_light', pos='0 0 4', mode='trackcom')

            ## TODO: Feb 12 01:33, add the multi ball rendering pipeline for paper vis.
            for i_ag in range(num_agents+1):
                if i_ag == 0:
                    name_bl = 'ball'
                    name_bl_root = 'ball_root'
                    name_bl_marker = 'ball_marker'
                    name_bl_light = f'ball_light'
                else:
                    name_bl = f'ball_copy_{i_ag}'
                    name_bl_root = f'ball_root_copy_{i_ag}'
                    name_bl_marker = f'ball_marker_copy_{i_ag}'
                    name_bl_light = f'ball_light_copy_{i_ag}'

                ball = ET.SubElement(worldbody, 'body', name=name_bl, pos='0 0 0.258') ## ori z=0.5
                
                if self.is_ball_freejnt:
                    ## just for vis for no need to fix it
                    ET.SubElement(ball, 'freejoint', name=name_bl_root)
                    conaffinity = '0'
                else:
                    conaffinity = '1'

                ET.SubElement(
                    ball,
                    'geom',
                    name=name_bl,
                    size='.25',
                    material='ball',
                    priority='1',
                    conaffinity=conaffinity, ## original: 1
                    condim='6',
                    ## by luo
                    # contype='0',
                    # conaffinity='0',
                    # density='0',
                )

                ## ------------
                ### for visulization only
                ET.SubElement(
                    worldbody, ## important
                    'geom',
                    name=name_bl_marker,
                    type='cylinder',
                    size='0.45 .05',
                    # type='sphere',
                    # size='.05',
                    pos='-10 0 .05',
                    # rgba='0.97 0.30 0. 0.78',
                    rgba='1. 0.96 0.1 0.78',
                    contype='0',
                    conaffinity='0',
                    density='0',
                    ##
                    # material='unlit_material',
                )

                ## Light for the ball
                # ET.SubElement(ball, 'light', name=name_bl_light, pos='0 0 4', mode='trackcom')


                ## ------------




            ## NEW Feb 4, for better vis
            ET.SubElement(
                worldbody,
                'geom',
                name='ball_start_marker', ## subgoal waypoint
                type='cylinder',
                size='0.7 .05',
                pos='-10 0 .05',
                rgba='0.97 0.30 0. 0.78',
                contype='0',
                conaffinity='0',
                density='0',
            )

        def set_tasks(self):
            # `tasks` is a list of tasks, where each task is a list of three tuples: (agent_init_ij, ball_init_ij,
            # goal_ij).
            if self._maze_type == 'arena':
                tasks = [
                    [(1, 6), (2, 3), (5, 2)],
                    [(2, 2), (5, 5), (2, 2)],
                    [(6, 1), (2, 3), (6, 6)],
                    [(6, 6), (1, 1), (6, 1)],
                    [(4, 6), (6, 2), (1, 6)],
                ]
            elif self._maze_type == 'medium':
                tasks = [
                    [(1, 1), (3, 4), (6, 6)],
                    [(6, 1), (6, 5), (1, 1)],
                    [(5, 3), (4, 2), (6, 5)],
                    [(6, 5), (1, 1), (5, 3)],
                    [(1, 6), (6, 1), (1, 6)],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.task_infos = []
            for i, task in enumerate(tasks):
                self.task_infos.append(
                    dict(
                        task_name=f'task{i + 1}',
                        agent_init_ij=task[0],
                        agent_init_xy=self.ij_to_xy(task[0]),
                        ball_init_ij=task[1],
                        ball_init_xy=self.ij_to_xy(task[1]),
                        goal_ij=task[2],
                        goal_xy=self.ij_to_xy(task[2]),
                    )
                )

        def reset(self, options=None, *args, **kwargs):
            if options is None:
                options = {}
            # Set the task goal.
            if 'task_id' in options:
                # Use the pre-defined task.
                assert 1 <= options['task_id'] <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = options['task_id']
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_info' in options:
                # Use the provided task information.
                self.cur_task_id = None
                self.cur_task_info = options['task_info']
            else:
                # Randomly sample a task.
                self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]

            # Whether to provide a rendering of the goal.
            render_goal = False
            if 'render_goal' in options:
                render_goal = options['render_goal']

            # Get initial and goal positions with noise.
            agent_init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['agent_init_ij']))
            ball_init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['ball_init_ij']))
            goal_xy = self.ij_to_xy(self.cur_task_info['goal_ij'])
            if self._add_noise_to_goal:
                goal_xy = self.add_noise(goal_xy)

            # First, force set the position to the goal position to obtain the goal observation.
            super(MazeEnv_Multi_Agent_Luo, self).reset(*args, **kwargs)

            # Do a few random steps to stabilize the environment.
            for _ in range(10):
                super(MazeEnv_Multi_Agent_Luo, self).step(self.action_space.sample())

            # Save the goal observation.
            self.set_goal(goal_xy=goal_xy)
            self.set_agent_ball_xy(goal_xy, goal_xy)
            goal_ob = self.get_ob()
            if render_goal:
                goal_rendered = self.render()

            # Now, do the actual reset.
            ob, info = super(MazeEnv_Multi_Agent_Luo, self).reset(*args, **kwargs)
            self.set_goal(goal_xy=goal_xy)
            self.set_agent_ball_xy(agent_init_xy, ball_init_xy)
            ob = self.get_ob()
            info['goal'] = goal_ob
            if render_goal:
                info['goal_rendered'] = goal_rendered

            return ob, info

        def step(self, action):
            ob, reward, terminated, truncated, info = super(MazeEnv_Multi_Agent_Luo, self).step(action)

            # Check if the ball has reached the goal.
            if np.linalg.norm(self.get_agent_ball_xy()[1] - self.cur_goal_xy) <= self._goal_tol:
                if self._terminate_at_goal:
                    terminated = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

            return ob, reward, terminated, truncated, info

        def get_agent_ball_xy(self):
            agent_xy = self.data.qpos[:2].copy()
            ball_xy = self.data.qpos[-7:-5].copy()

            return agent_xy, ball_xy

        def set_agent_ball_xy(self, agent_xy, ball_xy):
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            qpos[:2] = agent_xy
            qpos[-7:-5] = ball_xy
            self.set_state(qpos, qvel)


        def set_ball_start_marker(self, start_xy):
            ## visualize the start position
            if self._ob_type == 'states':
                self.model.geom('ball_start_marker').pos[:2] = start_xy
                mujoco.mj_forward(self.model, self.data)
            



    


    if maze_env_type == 'maze':
        return MazeEnv_Multi_Agent_Luo(*args, **kwargs)
    elif maze_env_type == 'ball':
        return BallEnv_Multi_Agent_Luo(*args, **kwargs)
    else:
        raise ValueError(f'Unknown maze environment type: {maze_env_type}')

