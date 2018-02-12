from gym.envs.registration import register
import HER

register(
    id='Baxter3dbox-v0',
    entry_point='HER.envs.baxter_orient_left_cts_3d_box_random_start:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='Baxter3dpen-v3',
    entry_point='HER.envs.baxter_orient_left_cts_3d_pen_random_start:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='Baxter3dpen-v2',
    entry_point='HER.envs.baxter_orient_left_cts_3d_pen_multiple_start:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='Baxter3dpen-v0',
    entry_point='HER.envs.baxter_orient_left_cts_3d_pen_simple:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='Baxter3dpen-v1',
    entry_point='HER.envs.baxter_orient_left_cts_3d_pen:BaxterEnv',
    kwargs={'max_len':50}
)


register(
    id='Baxter3d-v1',
    entry_point='HER.envs.baxter_orient_left_cts_3d:BaxterEnv',
    kwargs={'max_len':50}
)

###########
register(
    id='Baxter-v10',
    entry_point='HER.envs.baxter_orient_left_cts_near:BaxterEnv',
    kwargs={'max_len':30}
)

register(
    id='Baxter-v11',
    entry_point='HER.envs.baxter_orient_left_cts_near:BaxterEnv',
    kwargs={'max_len':20}
)

# Baxter envs
register(
    id='Baxter-v1',
    entry_point='HER.envs.baxter_orient_left_cts:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='Baxter-v8',
    entry_point='HER.envs.baxter_orient_left_cts:BaxterEnv',
    kwargs={'max_len':20}
)


register(
    id='Baxter-v9',
    entry_point='HER.envs.baxter_orient_left_cts:BaxterEnv',
    kwargs={'max_len':30}
)

register(
    id='Baxter-v2',
    entry_point='HER.envs.baxter_orient_left_cts_angle_reward:BaxterEnv',
    # More arguments here
)

register(
    id='Baxter-v3',
    entry_point='HER.envs.baxter_orient_left_cts_bounds:BaxterEnv',
    # More arguments here
)

register(
    id='Baxter-v4',
    entry_point='HER.envs.baxter_orient_left_cts_azimuth:BaxterEnv',
    # More arguments here
)

register(
    id='Baxter-v5',
    entry_point='HER.envs.baxter_orient_left_cts_euler:BaxterEnv',
    # More arguments here
)

register(
    id='Baxter-v6',
    entry_point='HER.envs.baxter_orient_left_cts_anglebw:BaxterEnv',
    # More arguments here
)

register(
    id='Baxter-v7',
    entry_point='HER.envs.baxter_orient_left_cts_onetarget:BaxterEnv',
    # More arguments here
)

register(
    id='BaxterReacher-v1',
    entry_point='HER.envs.baxter_orient_left_reacher:BaxterEnv',
    kwargs={"max_len":50}
)

register(
    id='BaxterReacher-v3',
    entry_point='HER.envs.baxter_orient_left_reacher:BaxterEnv',
    kwargs={"max_len":10}
)


register(
    id='BaxterReacher-v2',
    entry_point='HER.envs.baxter_orient_left_reacher_shaped:BaxterEnv',
    # More arguments here
)
