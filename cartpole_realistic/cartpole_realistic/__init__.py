from gym.envs.registration import register

register(
    id='cartpole-realistic',
    entry_point='cartpole_realistic.envs:CartPoleEnv',
    max_episode_steps=500,
)
