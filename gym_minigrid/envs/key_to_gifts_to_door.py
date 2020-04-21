import gym
from gym_minigrid.register import register

from gym_minigrid.envs.fetchkey import KeyEnv
from gym_minigrid.envs.opengifts import GiftsEnv
from gym_minigrid.envs.doorkeyoptional import DoorKeyOptionalEnv


class KeyToGiftsToDoorKeyOptional(gym.Wrapper):
    """
    Stitches together three environments: 
    (1) KeyEnv 
    (2) GiftsEnv
    (3) DoorKeyOptionalEnv

    Agent is teleported from (1) to (2) to (3).  The agent is initialized with or
    without the key in (3), depending on whether it successfully fetched it in (1).
    """
    def __init__(self, key_kwargs, gifts_kwargs, doorkeyoptional_kwargs):
        self._envs = [KeyEnv, GiftsEnv, DoorKeyOptionalEnv]
        self._env_kwargs = [key_kwargs, gifts_kwargs, doorkeyoptional_kwargs]
        self.num_phases = len(self._envs)
        self._env_idx = None  # index of the current env
        self.env = KeyEnv(**key_kwargs)
        super().__init__(self.env)


    def reset(self):
        """reset returns agent back to first environment, KeyEnv"""
        self._env_idx = 0
        self.env = self._envs[self._env_idx](**self._env_kwargs[self._env_idx])
        observation = self.env.reset()
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done is True and self._env_idx < self.num_phases - 1:
            if self._env_idx == 0:
                # if agent fetched key in first environment, initialize it with a key
                # in the last environment
                self._env_kwargs[2]['key_color'] = info.get('carrying_key_color')

            # teleport to the next environment
            self._env_idx += 1
            self.env = self._envs[self._env_idx](**self._env_kwargs[self._env_idx])
            observation, reward, done, info = self.env.reset(), 0, False, {}

        return observation, reward, done, info


class TinyKeyGiftsDoorEnv(KeyToGiftsToDoorKeyOptional):
    def __init__(self):
        key_kwargs = dict(
            size=6,
            key_color='yellow',
            start_by_key=False,
            max_steps=10
        )
        gifts_kwargs = dict(
            size=6,
            num_objs=3,
            gift_reward=1,
            max_steps=5*6**2
        )
        doorkeyoptional_kwargs = dict(
            size=8,
            key_color=None,
            door_color='yellow',
            max_steps=30
        )
        super().__init__(key_kwargs, gifts_kwargs, doorkeyoptional_kwargs)


register(
    id='MiniGrid-KeyGiftsDoor-tiny-v0',
    entry_point='gym_minigrid.envs:TinyKeyGiftsDoorEnv'
)
