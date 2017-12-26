from gym.envs.registration import register
import HER

# Baxter envs
register(
    id='Baxter-v1',
    entry_point='HER.envs.baxter_orient_left_cts:BaxterEnv',
    # More arguments here
)
