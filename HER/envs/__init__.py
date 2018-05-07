from gym.envs.registration import register
import HER

# Fetch envs
register(
    id='release-v0',
    entry_point='HER.envs.release_withextras:BaxterEnv',
    kwargs={'max_len':10}
)

register(
    id='fetchpnp-v0',
    entry_point='HER.envs.fetchpnp:FetchPnp',
    kwargs={'reward_type':''},
    max_episode_steps=50,
)

# Baxter envs
register(
    id='putainb-v0',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='putainbt-v0',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':50,
            'test':True}
)

register(
    id='Reacher2d-v0',
    entry_point='HER.envs.reacher2d:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='Reacher2d-v1',
    entry_point='HER.envs.reacher2d_rel:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='Reacher3d-v0',
    entry_point='HER.envs.reacher3d:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='pusher-v0',
    entry_point='HER.envs.pusher:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='pusher-v1',
    entry_point='HER.envs.close_pusher:BaxterEnv',
    kwargs={'max_len':10}
)

register(
    id='pusher-v2',
    entry_point='HER.envs.close_pusher_rel:BaxterEnv',
    kwargs={'max_len':10}
)

register(
    id='grasping-v0',
    entry_point='HER.envs.grasping:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='grasping-v1',
    entry_point='HER.envs.grasping_rel:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='grasping-v2',
    entry_point='HER.envs.grasping_withgap:BaxterEnv',
    kwargs={'max_len':20}
)
register(
    id='graspingt-v2',
    entry_point='HER.envs.grasping_withgap:BaxterEnv',
    kwargs={'max_len':20, 'test':True}
)

register(
    id='picknmove-v0',
    entry_point='HER.envs.picknmove:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='picknmove-v1',
    entry_point='HER.envs.picknmove_rel:BaxterEnv',
    kwargs={'max_len':50}
)


register(
    id='picknmove-v2',
    entry_point='HER.envs.picknmove_withextras:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='picknmovet-v2',
    entry_point='HER.envs.picknmove_withextras:BaxterEnv',
    kwargs={'max_len':50, 'test':True}
)

register(
    id='picknmoved-v2',
    entry_point='HER.envs.picknmovedense_withextras:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='picknmovedt-v2',
    entry_point='HER.envs.picknmovedense_withextras:BaxterEnv',
    kwargs={'max_len':50, 'test':True}
)

register(
    id='picknmove-v3',
    entry_point='HER.envs.picknmove_withextras:BaxterEnv',
    kwargs={'max_len':100}
)

register(
    id='picknmovet-v3',
    entry_point='HER.envs.picknmove_withextras:BaxterEnv',
    kwargs={'max_len':100, 'test':True}
)