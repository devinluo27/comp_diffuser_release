

def get_gym_robo_env_name(env_name_ori):
    '''
    The input is our d4rl env_name 
    while we would like to get the corresponding env_name in gym_robot package, 
    for Ben's Maze2D and potentially AntMaze
    '''
    if "maze2d-umaze-v1" == env_name_ori: ## in
        env_name = "PointMaze_UMaze-v3"
    elif "maze2d-medium-v1" == env_name_ori:
        env_name = "PointMaze_Medium-v3"
    elif "maze2d-large-v1" == env_name_ori:
        env_name = "PointMaze_Large-v3"
    else:
        raise NotImplementedError

    return env_name

