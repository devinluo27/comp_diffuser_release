"""
Jan 20
We add multiple ants to the env
"""

import xml.etree.ElementTree as ET
import mujoco
import numpy as np

# def create_multi_ant_environment(xml_path, ant_positions):
def ogb_vis_create_multi_ant_environment(tree, num_agents):
    """
    Loads an existing MuJoCo XML (containing one Ant),
    then duplicates the Ant body at the specified positions.

    :param xml_path: Path to the base .xml file (e.g., 'ant_maze.xml')
    :param ant_positions: List of (x, y, z) positions where new Ants will be placed
    :return: A MuJoCo model (MjModel) with multiple Ants
    """
    # Parse the XML
    # tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the <worldbody> element
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("No <worldbody> found in the XML.")

    # Find the first <body> in the worldbody. 
    # Adjust this if your file structure is different or has multiple bodies.
    ant_body = worldbody.find('body')
    if ant_body is None:
        raise ValueError("Could not find an Ant <body> to duplicate.")
    


    ## NOTE:
    print(f"{ant_body}, {worldbody.find('body').findall(f'.//light')=}")
    ## ----------------------- Jan 29 ---------------------
    ## remove some elements, e.g., the light from the orginal ant,
    ## so all ants do not have light
    for tmp_obj in ant_body.findall(f'.//light'):
        tmp_obj_name = tmp_obj.get('name')
        
        # print(f'{i=} {tmp_obj_name=}')

        if tmp_obj_name == 'torso_light':
            ant_body.remove(tmp_obj)
    ## ------------------------------------------------------
    

    ## ------ Mar 2, add/copy other gadets  ------

    for i, i_ag in enumerate(range(num_agents), start=1):
        obj_name = 'marker_1'
        obj_name_sh = f".//geom[@name='{obj_name}']"
        # obj_name = ".//*[@name='marker_1']"
        tmp_obj = worldbody.find(obj_name_sh)
        # print(f'{tmp_obj=}')
        new_obj = ET.fromstring(ET.tostring(tmp_obj))
        new_obj.set('name', f'{obj_name}_copy_{i}')
        new_obj.set('pos', f"-10 {-10-i_ag/10} 0.012")
        worldbody.append(new_obj)

    ## -------------------------------------------



    print(f"{ant_body}, {worldbody.find('body').findall(f'.//light')=}")
    # We assume 'ant_body' is the entire Ant. We'll keep the original Ant and just duplicate it.

    # For each position, copy the Ant body, rename it, and shift it.
    # If you want N total ants, but already have 1, you'll create N-1 new ones, etc.
    # for i, pos in enumerate(ant_positions, start=1):
    for i, i_ag in enumerate(range(num_agents), start=1):
        # Make a deep copy of the Ant body
        new_ant = ET.fromstring(ET.tostring(ant_body))
        
        # Update the name to avoid collisions (e.g., ant, ant_1, ant_2, etc.)
        new_ant.set('name', f'ant_{i}')
        
        # pos = np.random.random(size=(3,)) * 20 ## NOTE: before Feb 10
        ## added on Feb 10 , so the rendering camera position will be the same as default!
        pos = [10, 10, 0.75]

        # Shift its position
        new_ant.set('pos', f"{pos[0]} {pos[1]} {pos[2]}")

        ## remove some elements
        for tmp_obj in new_ant.findall(f'.//light'):
            tmp_obj_name = tmp_obj.get('name')
            
            # print(f'{i=} {tmp_obj_name=}')

            if tmp_obj_name == 'torso_light':
                new_ant.remove(tmp_obj)


        
        # Optionally, rename sub-elements (geoms, joints, etc.) to avoid collisions 
        # if needed. For a minimal example, we skip that here.


        ### ----------------------------------------------
        ### Rename everthing
        ### ----------------------------------------------
        rename_list = ['geom', 'joint', 'camera', 'light', 'body']
        for elem_name in rename_list:
            for tmp_obj in new_ant.findall(f'.//{elem_name}'):
                tmp_obj_name = tmp_obj.get('name')
                if tmp_obj_name:
                    tmp_new_name = f"{tmp_obj_name}_copy_{i}"
                    tmp_obj.set('name', tmp_new_name)
                    # print(f'{tmp_obj_name=}, {tmp_new_name=}')


        # for geom in new_ant.findall('.//geom'):
        #     geom_name = geom.get('name')
        #     if geom_name:
        #         geom.set('name', f"{geom_name}_copy_{i}")

        # for joint in new_ant.findall('.//joint'):
        #     joint_name = joint.get('name')
        #     if joint_name:
        #         joint.set('name', f"{joint_name}_copy_{i}")

        


        ### ----------------------------------------------
        ### ----------------------------------------------
        

        # Append the new ant to the worldbody
        worldbody.append(new_ant)
    
    return tree



def find_subarray_index(value, sub_arrays):
    for i, sub_arr in enumerate(sub_arrays):
        if value in sub_arr:
            return i
    assert False


def get_sub_chunk_color(i_ag, num_agents, n_comp):
    import matplotlib.colors as mcolors

    from diffuser.guides.render_utils import Sub_Sg_Colors, Sub_Traj_Alpha

    Sub_Traj_Alpha = [0.3, 0.4, 0.28, 0.4, 0.3, 
                  0.45, 0.4, 0.4, 0.4, 0.4, 
                  0.4, 0.4, 0.4]
    Sub_Sg_Colors[5] = 'olivedrab' # aquamarine, orchid, peru, seems ok: seagreen
    # print(Sub_Sg_Colors)

    # per_color_len = num_agents // n_comp
    # np.array_split(np.arange(num_agents), n_comp)
    tmp_list = np.array_split(np.arange(num_agents), n_comp)
    c_idx = find_subarray_index(i_ag, tmp_list)
    # i_ag // per_color_len
    c_name = Sub_Sg_Colors[c_idx]
    tmp_alp = Sub_Traj_Alpha[c_idx]
    # print(tmp_list)
    # print(c_name)
    tmp_rgba = list(mcolors.to_rgb(c_name) ) + [tmp_alp,]
    return tmp_rgba


def set_env_state_multi_agent(env_ag, num_agents, full_states, n_comp):
    '''
    Feb 2
    Set the state of each ant in the multi-ant env
    '''
    import matplotlib
    # cmap_name = 'Oranges'
    cmap_name = 'Reds'
    cmap = matplotlib.colormaps.get_cmap('viridis')
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    
    wp_alpha = 0.3
    # wp_alpha = 0.5
    # wp_alpha = 1.0

    assert len(full_states) == num_agents + 1
    qpos = env_ag.data.qpos.copy()
    print(f'{qpos.shape}')
    len_qpos = 15

    for i_ag in range(num_agents+1):
    # for i_ag in range(num_agents-2):
        if i_ag % 20 == 0:
            print(f'{i_ag=} {i_ag*len_qpos}:{(i_ag+1)*len_qpos}, {full_states[i_ag][:2]}')
            # print(f'{i_ag=} {i_ag*len_qpos}:{(i_ag+1)*len_qpos}')

        qpos[i_ag*len_qpos:(i_ag+1)*len_qpos] = full_states[i_ag][:len_qpos]
        # print(i_ag / len(full_states))
        if False:
            tmp_rgba = np.array( cmap( i_ag / len(full_states) ) )
        else:
            tmp_rgba = get_sub_chunk_color(i_ag, num_agents+1, n_comp)

        # tmp_rgba[-1] = wp_alpha
        # print(f'{tmp_rgba=}')

        if i_ag == 0:
            vc_name = f'visual_circle'
        else:
            vc_name = f'visual_circle_copy_{i_ag}'
        # print(tmp_rgba)
        env_ag.model.geom(vc_name).rgba = tmp_rgba # np.array([0. , 1. , 0. , 0.1])
        tmp_vc_pos = env_ag.model.geom(vc_name).pos
        tmp_vc_pos[2] = -0.01 # - 0.1

    ## set the whole env_ag qpos for all ants
    env_ag.data.qpos = qpos
    mujoco.mj_forward(env_ag.model, env_ag.data)
    for i in range(1):
        # mujoco.mj_forward(env.model, env.data)
        env_ag.step(np.zeros(shape=(8,)))


# def set_all_visual_circles_alpha



import seaborn as sns

def get_sub_chunk_color_v2(i_ag, num_agents, n_comp, c_palette, sub_tj_alpha):
    import matplotlib.colors as mcolors

    # from diffuser.guides.render_utils import Sub_Sg_Colors, Sub_Traj_Alpha

    # Sub_Traj_Alpha = [0.3, 0.4, 0.28, 0.4, 0.3, 
    #             0.45, 0.4, 0.4, 0.4, 0.4, 
    #             0.4, 0.4, 0.4]
    # Sub_Sg_Colors[5] = 'olivedrab' # aquamarine, orchid, peru, seems ok: seagreen
    # print(Sub_Sg_Colors)
    Sub_Traj_Alpha = [0.5] * 10
    Sub_Traj_Alpha = [0.45] * 10

    # per_color_len = num_agents // n_comp
    # np.array_split(np.arange(num_agents), n_comp)
    tmp_list = np.array_split(np.arange(num_agents), n_comp)
    c_idx = find_subarray_index(i_ag, tmp_list)
    # i_ag // per_color_len
    ## actuall a tuple of len 3, [0,1]
    
    use_color = sns.color_palette('muted')[c_idx]
    if c_palette is not None:
        use_color = c_palette[c_idx]
    
    tmp_alp = Sub_Traj_Alpha[c_idx]
    if sub_tj_alpha is not None:
        tmp_alp = sub_tj_alpha[c_idx]
    
    # print(tmp_list)
    # print(c_palette)
    tmp_rgba = list(mcolors.to_rgb(use_color) ) + [tmp_alp,]
    return tmp_rgba


def set_env_state_multi_agent_v2(env_ag, num_agents, full_states, n_comp, 
                                 c_palette=None,
                                 sub_tj_alpha=None):
    '''
    Feb 2
    Set the state of each ant in the multi-ant env
    '''
    import matplotlib
    # cmap_name = 'Oranges'
    cmap_name = 'Reds'
    cmap = matplotlib.colormaps.get_cmap('viridis')
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    
    wp_alpha = 0.3
    # wp_alpha = 0.5
    # wp_alpha = 1.0

    assert len(full_states) == num_agents + 1
    qpos = env_ag.data.qpos.copy()
    print(f'{qpos.shape}')
    # if len(env_ag.data.qpos) == 17:
    if False:
        len_qpos = 17
    else:
        len_qpos = 15

    for i_ag in range(num_agents+1):
    # for i_ag in range(num_agents-2):
        if i_ag % 20 == 0:
            print(f'{i_ag=} {i_ag*len_qpos}:{(i_ag+1)*len_qpos}, {full_states[i_ag][:2]}')
            # print(f'{i_ag=} {i_ag*len_qpos}:{(i_ag+1)*len_qpos}')

        qpos[i_ag*len_qpos:(i_ag+1)*len_qpos] = full_states[i_ag][:len_qpos]
        # print(i_ag / len(full_states))
        if False:
            tmp_rgba = np.array( cmap( i_ag / len(full_states) ) )
        else:
            # tmp_rgba = get_sub_chunk_color(i_ag, num_agents+1, n_comp)
            # tmp_rgba = sns.color_palette('muted')

            
            tmp_rgba = get_sub_chunk_color_v2(i_ag, num_agents+1, n_comp, c_palette, sub_tj_alpha)
            
            

        # tmp_rgba[-1] = wp_alpha
        # print(f'{tmp_rgba=}')

        if i_ag == 0:
            vc_name = f'visual_circle'
        else:
            vc_name = f'visual_circle_copy_{i_ag}'
        # print(tmp_rgba)
        env_ag.model.geom(vc_name).rgba = tmp_rgba # np.array([0. , 1. , 0. , 0.1])
        tmp_vc_pos = env_ag.model.geom(vc_name).pos
        tmp_vc_pos[2] = -0.01 # - 0.1

    ## set the whole env_ag qpos for all ants
    env_ag.data.qpos = qpos
    mujoco.mj_forward(env_ag.model, env_ag.data)
    for i in range(1):
        # mujoco.mj_forward(env.model, env.data)
        env_ag.step(np.zeros(shape=(8,)))



def set_env_state_multi_agent_v2_Soccer(env_ag, num_agents, full_states, n_comp, 
                                 c_palette=None,
                                 sub_tj_alpha=None, use_ball_mk=False):
    '''
    Feb 2
    Set the state of each ant in the multi-ant env
    '''
    import matplotlib
    # cmap_name = 'Oranges'
    cmap_name = 'Reds'
    cmap = matplotlib.colormaps.get_cmap('viridis')
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    
    wp_alpha = 0.3
    # wp_alpha = 0.5
    # wp_alpha = 1.0

    assert len(full_states) == num_agents + 1
    qpos = env_ag.data.qpos.copy()
    print(f'{qpos.shape}')
    # if len(env_ag.data.qpos) == 17:
    if False:
        len_qpos = 17
    else:
        len_qpos = 15
    len_ant_qpos = (num_agents+1) * len_qpos

    for i_ag in range(num_agents+1):
    # for i_ag in range(num_agents-2):
        if i_ag % 20 == 0:
            print(f'{i_ag=} {i_ag*len_qpos}:{(i_ag+1)*len_qpos}, {full_states[i_ag][:2]}')
            # print(f'{i_ag=} {i_ag*len_qpos}:{(i_ag+1)*len_qpos}')

        qpos[i_ag*len_qpos:(i_ag+1)*len_qpos] = full_states[i_ag][:len_qpos]
        
        name_ball = f'ball_copy_{i_ag}' if i_ag > 0 else 'ball'
        env_ag.model.body(name_ball).pos[:2] = full_states[i_ag][len_qpos:]
        if len(env_ag.data.qpos) > len_ant_qpos: ## 15
            qpos[ len_ant_qpos + i_ag*7 : len_ant_qpos + i_ag*7 + 2 ] = full_states[i_ag][len_qpos:]
        # print(i_ag / len(full_states))
        if False:
            tmp_rgba = np.array( cmap( i_ag / len(full_states) ) )
        else:
            # tmp_rgba = get_sub_chunk_color(i_ag, num_agents+1, n_comp)
            # tmp_rgba = sns.color_palette('muted')

            
            tmp_rgba = get_sub_chunk_color_v2(i_ag, num_agents+1, n_comp, c_palette, sub_tj_alpha)
            
            

        # tmp_rgba[-1] = wp_alpha
        # print(f'{tmp_rgba=}')

        if i_ag == 0:
            vc_name = f'visual_circle'
            name_bl_marker = 'ball_marker'
        else:
            vc_name = f'visual_circle_copy_{i_ag}'
            name_bl_marker = f'ball_marker_copy_{i_ag}'

        # print(tmp_rgba)
        env_ag.model.geom(vc_name).rgba = tmp_rgba # np.array([0. , 1. , 0. , 0.1])
        tmp_vc_pos = env_ag.model.geom(vc_name).pos
        tmp_vc_pos[2] = -0.01 # - 0.1
        
        if use_ball_mk:
            env_ag.model.geom(name_bl_marker).pos[:2] = full_states[i_ag][len_qpos:]



    ## set the whole env_ag qpos for all ants
    env_ag.data.qpos = qpos
    mujoco.mj_forward(env_ag.model, env_ag.data)
    for i in range(1):
        # mujoco.mj_forward(env.model, env.data)
        env_ag.step(np.zeros(shape=(8,)))


