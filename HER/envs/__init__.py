from gym.envs.registration import register
import HER

# Baxter envs
register(
    id='Baxter-v1',
    entry_point='HER.envs.baxter_orient_left_cts:BaxterEnv',
    # More arguments here
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
    # More arguments here
)


register(
    id='BaxterReacher-v2',
    entry_point='HER.envs.baxter_orient_left_reacher_shaped:BaxterEnv',
    # More arguments here
)
