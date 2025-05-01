import matplotlib.pyplot as plt
import numpy as np
import pdb
import matplotlib.colors as colors
Extra_Colors = ["aquamarine", "springgreen", "slateblue", "tomato", "hotpink", "yellow", "peru", "cyan",]
Extra_Alpha = [0.4,] * len(Extra_Colors)
# Sub_Traj_Colors = ["Blues", "Greys", "Oranges", "Purples", "Greens", "pink",] + Extra_Colors 
# Sub_Sg_Colors = ["blue", "grey", "orange", "purple", "green", "pink"] + Extra_Colors
# Sub_Traj_Alpha = [0.55, 0.5, 0.35, 0.4, 0.5, 0.4] + Extra_Alpha
## v2 Jan 22, change color order, "Purples",
Sub_Traj_Colors = ["Blues", "Greens", "Oranges", "Purples", "Greys", ] + Extra_Colors 
Sub_Sg_Colors = ["blue", "green", "orange", "purple", "grey", ] + Extra_Colors
# Sub_Traj_Alpha = [0.55, 0.6, 0.28, 0.30, 0.4, ] + Extra_Alpha ## purple ori: 0.3;
Sub_Traj_Alpha = [0.55, 0.6, 0.3, 0.32, 0.35, ] + Extra_Alpha ## Feb 4


'''From the Diffusion Forcing Repo'''

def is_grid_env(env_id):
    return "maze2d" in env_id or "diagonal2d" in env_id

def get_maze_grid_from_env(env):
    maze_string = env.str_maze_spec
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]

def get_maze_grid_from_id(env_id):
    import gym

    maze_string = gym.make(env_id).str_maze_spec
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def plot_maze_layout(ax, maze_grid):
    ax.clear()

    if maze_grid is not None:
        for i, row in enumerate(maze_grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                    ax.add_patch(square)

    ax.set_aspect("equal")
    ax.grid(True, color="white", linewidth=4)
    ax.set_axisbelow(True)
    if len(maze_grid) >= 5:
        tmp_line_w = 4
    else:
        tmp_line_w = 2

    ax.spines["top"].set_linewidth(tmp_line_w)
    ax.spines["right"].set_linewidth(tmp_line_w)
    ax.spines["bottom"].set_linewidth(tmp_line_w)
    ax.spines["left"].set_linewidth(tmp_line_w)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
    ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
    ax.set_xlim(0.5, len(maze_grid) + 0.5)
    ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
    ax.grid(True, color="white", which="minor", linewidth=4)


def plot_start_goal(ax, start_goal: None, sg_color="black", 
                    size_out_cicle=0.26,
                    size_in_st=0.135,
                    size_in_goal=0.17,):
    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)

    ## hyper-param for scale 0.6, should be larger if scale is larger
    # size_out_cicle = 0.26
    # size_in_st = 0.135
    # size_in_goal = 0.17

    start_x, start_y = start_goal[0]
    start_outer_circle = plt.Circle((start_x, start_y), size_out_cicle, facecolor="white", edgecolor="black")
    ax.add_patch(start_outer_circle)
    start_inner_circle = plt.Circle((start_x, start_y), size_in_st, color=sg_color) # black
    ax.add_patch(start_inner_circle)

    goal_x, goal_y = start_goal[1]
    goal_outer_circle = plt.Circle((goal_x, goal_y), size_out_cicle, facecolor="white", edgecolor="black")
    ax.add_patch(goal_outer_circle)
    draw_star((goal_x, goal_y), radius=size_in_goal, color=sg_color)

# MAZE_PLOT_BG_SIZE = {}

def make_traj_images(env, trajs: np.ndarray, start=None, goal=None, 
                is_non_keypt=None, bg_scale=0.6,
                plot_end_points=True, batch_first=True, env_id=None, titles=None,
                sp_pts_1=None,
                sp_xy_1=None,
                sp_xy_2=None,
                trajs_2=None, trajs_3=None, trajs_4=None, trajs_list=None, sp_xy_3=None, sp_xy_4=None, trajs_ball:list=None, fig_dpi=100, main_tj_cmap='Reds', is_plot_main_tj=True, save_pdf_path=None,
                sub_tj_colors=None, 
                sub_tj_alpha=None,
                sc_size=None, ## size for the scatter plot marker
                ):
    '''
    Give a batch of trajs (only xy), return a batch of images (H,W,4) in a list
        env: the actual env instance
        start (np 2d): [B,dim(2)]
        env_id (str): name of the env
        is_non_keypt (np bool): B,H
        sp_pts_1: special pts idxs
        trajs_ball: a list of 2d trajs of the ball in ant soccer, therefore support different len,
                    color similar to trajs_2
        trajs_list: a list of batch trajs to be in one image, np4d or list[np3d]; Shape: n_tj,B,hzn,d
    TODO: In the DF code, the input trajs are in shape (H, B, Dim)
    '''

    assert type(trajs) == np.ndarray and trajs.ndim == 3 and trajs.shape[2] in [4,2]
    if trajs.shape[2] == 4: ## convert to xy only
        trajs = trajs[:, :, :2]

    ## trajs must be in shape (H, B, Dim)
    # if trajs.shape[0] == batch_size:
    if batch_first:
        ## B,H,dim to H,B,dim
        trajs = np.transpose(trajs, (1,0,2))
    
    # pdb.set_trace()
    batch_size = trajs.shape[1]

    if start is None:
        start = trajs[0, :, :].copy()
    if goal is None:
        goal = trajs[-1, :, :].copy()
    else:
        assert goal.shape == (batch_size,2)
    
    images = []
    for batch_idx in range(batch_size):
        fig, ax = plt.subplots(dpi=fig_dpi)

        if env_id is None: ## use env
            maze_grid = get_maze_grid_from_env(env)
        else:
            if is_grid_env(env_id):
                maze_grid = get_maze_grid_from_id(env_id)
            else:
                maze_grid = None

        ## important: luo update 13:22, July23
        f_h = len(maze_grid[0]) * bg_scale
        f_w = len(maze_grid) * bg_scale
        fig.set_figheight(  f_h )
        fig.set_figwidth( f_w )
        ## ---------
        
        plot_maze_layout(ax, maze_grid)
        if is_non_keypt is None:
            ## no need to highlight keypt
            if is_plot_main_tj:
                ax.scatter(trajs[:, batch_idx, 0], trajs[:, batch_idx, 1], c=np.arange(len(trajs)), cmap=main_tj_cmap, s=sc_size)
        else:
            is_non_k = is_non_keypt[batch_idx]
            c_map = np.arange(len(trajs)) ## 1d (H,)
            ## plot key
            plt.scatter(trajs[is_non_k, batch_idx, 0], trajs[is_non_k, batch_idx, 1], 
                                    c=c_map[is_non_k], cmap="Reds")
                            # c=colors[is_non_keypt], zorder=20, alpha=0.3, s=40) # s not given
            ## plot key
            is_k = ~ is_non_k
            plt.scatter(trajs[is_k, batch_idx, 0], trajs[is_k, batch_idx, 1], 
                            c=c_map[is_k], cmap="Greens", marker='*', s=70) #s=400, zorder=25, 
            
            if sp_pts_1 is not None:
                sp_1_tmp = sp_pts_1[batch_idx]
                plt.scatter(trajs[sp_1_tmp, batch_idx, 0], trajs[sp_1_tmp, batch_idx, 1], 
                            c=c_map[sp_1_tmp], cmap="Blues", marker='^', s=180) #s=400, zorder=25, 
        
        #### -------
        if sp_xy_1 is not None: ## B, 2
            # pdb.set_trace()
            plt.scatter(sp_xy_1[batch_idx, :, 0], sp_xy_1[batch_idx, :, 1], 
                        c=np.arange(sp_xy_1.shape[1]), 
                        cmap="Greys", marker='>', s=50, alpha=0.3) #s=400, zorder=25, Oranges
            
        if sp_xy_2 is not None: ## B, 2
            plt.scatter(sp_xy_2[batch_idx, :, 0], sp_xy_2[batch_idx, :, 1], 
                        c=np.arange(sp_xy_2.shape[1]),
                        cmap="Purples", marker='>', s=50, alpha=0.3) #s=400, zorder=25, 


        if sp_xy_3 is not None: ## B, 2
            # pdb.set_trace()
            plt.scatter(sp_xy_3[batch_idx, :, 0], sp_xy_3[batch_idx, :, 1], 
                        c=np.arange(sp_xy_3.shape[1]), 
                        cmap="Oranges", marker='>', s=50, alpha=0.3) #s=400, zorder=25,
        
        if sp_xy_4 is not None: ## B, 2
            # pdb.set_trace()
            plt.scatter(sp_xy_4[batch_idx, :, 0], sp_xy_4[batch_idx, :, 1], 
                        c=np.arange(sp_xy_4.shape[1]), 
                        cmap="Greens", marker='>', s=50, alpha=0.3) #s=400, zorder=25,


        if trajs_ball is not None: ## B, 2
            plt.scatter(trajs_ball[batch_idx][:, 0], trajs_ball[batch_idx][:, 1], 
                        c=np.arange(trajs_ball[batch_idx].shape[0]),
                        cmap="Blues", marker='>', s=25, alpha=0.5) # s=50, 0.5
            start_goal_ball = ( trajs_ball[batch_idx][0, :], trajs_ball[batch_idx][-1, :], )
            plot_start_goal(ax, start_goal_ball, sg_color="blue",
                            size_out_cicle=0.22,
                            size_in_st=0.11,
                            size_in_goal=0.13,)



        if trajs_2 is not None: ## B, 2
            plt.scatter(trajs_2[batch_idx, :, 0], trajs_2[batch_idx, :, 1], 
                        c=np.arange(trajs_2.shape[1]),
                        cmap="Blues", marker='>', s=50, alpha=0.5) ## prev:0.25, viridis
            start_goal_2 = ( trajs_2[batch_idx, 0, :], trajs_2[batch_idx, -1, :], )
            plot_start_goal(ax, start_goal_2, sg_color="blue")
        if trajs_3 is not None:
            plt.scatter(trajs_3[batch_idx, :, 0], trajs_3[batch_idx, :, 1], 
                        c=np.arange(trajs_3.shape[1]),
                        cmap="Greys", marker='>', s=50, alpha=0.3) #s=400, zorder=25, 

            start_goal_3 = ( trajs_3[batch_idx, 0, :], trajs_3[batch_idx, -1, :], )
            plot_start_goal(ax, start_goal_3, sg_color="grey")

        if trajs_4 is not None:
            plt.scatter(trajs_4[batch_idx, :, 0], trajs_4[batch_idx, :, 1], 
                        c=np.arange(trajs_4.shape[1]),
                        cmap="Oranges", marker='>', s=50, alpha=0.3) #s=400, zorder=25, 

            start_goal_4 = ( trajs_4[batch_idx, 0, :], trajs_4[batch_idx, -1, :], )
            plot_start_goal(ax, start_goal_4, sg_color="orange")

        if trajs_list is not None:
            
            num_tj = len(trajs_list)
            assert num_tj <= len(Sub_Traj_Colors), f'{num_tj=}, smaller than the prepared colors'
            for i_tj, trajs_p in enumerate(trajs_list):
                if sub_tj_colors == None:
                    tmp_color_1 = Sub_Traj_Colors[i_tj]
                    tmp_color_2 = Sub_Sg_Colors[i_tj]
                    tmp_color_a = Sub_Traj_Alpha[i_tj]
                else:
                    tmp_color_1 = sub_tj_colors[i_tj]
                    tmp_color_2 = tmp_color_1
                    tmp_color_a = sub_tj_alpha[i_tj]
                    # assert tmp_color_1.islower()

                if sc_size is None:
                    tj_sc_size = 50 ## default 50 as in Feb 27
                else:
                    tj_sc_size = sc_size
                # if i_tj < 6:
                # if i_tj < 5:
                if isinstance(tmp_color_1, str) and tmp_color_1[0].isupper():
                    c_norm = colors.Normalize(vmin=-.5, vmax=1)  # Start colormap from 20 instead of 0
                    plt.scatter(trajs_p[batch_idx, :, 0], trajs_p[batch_idx, :, 1], 
                        c=np.linspace(0,1,trajs_p.shape[1]),
                        cmap=tmp_color_1, s=tj_sc_size, alpha=tmp_color_a, norm=c_norm) ## marker='>',
                else:
                    plt.scatter(trajs_p[batch_idx, :, 0], trajs_p[batch_idx, :, 1], 
                        c=tmp_color_1, s=tj_sc_size, alpha=tmp_color_a)


                tmp_start_goal = ( trajs_p[batch_idx, 0, :], trajs_p[batch_idx, -1, :], )
                plot_start_goal(ax, tmp_start_goal, sg_color=tmp_color_2)




        if plot_end_points and is_plot_main_tj: ## plot statt and goal
            start_goal = (start[batch_idx], goal[batch_idx])
            plot_start_goal(ax, start_goal)
        
        if titles is not None:
            plt.title(f"{titles[batch_idx]}", y=1.025, fontsize=25,) ## color='red', more space pad=10


        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)
        images.append(img)
        if save_pdf_path is not None:
            plt.savefig(save_pdf_path, format="pdf")
            print(f'{save_pdf_path=}')


        plt.close()
    return images