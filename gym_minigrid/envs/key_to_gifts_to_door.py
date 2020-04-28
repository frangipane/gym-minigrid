import gym
from gym_minigrid.register import register

from gym_minigrid.envs.fetchkey import KeyEnv
from gym_minigrid.envs.opengifts import GiftsEnv
from gym_minigrid.envs.doorkeyoptional import DoorKeyOptionalEnv


class KeyToGiftsToDoorKeyOptional(gym.Env):
    """
    Stitches together three environments: 
    (1) KeyEnv 
    (2) GiftsEnv -- this phase is optional
    (3) DoorKeyOptionalEnv

    Agent is teleported from (1) to (2) to (3).  The agent is initialized with or
    without the key in (3), depending on whether it successfully fetched it in (1).

    If the gifts_kwargs dictionary is empty, then omit the GiftsEnv.
    """
    def __init__(self, key_kwargs, gifts_kwargs, doorkeyoptional_kwargs, seed=111):
        if len(gifts_kwargs) == 0:
            self._envs = [KeyEnv, DoorKeyOptionalEnv]
            self._env_kwargs = [key_kwargs, doorkeyoptional_kwargs]
        else:
            self._envs = [KeyEnv, GiftsEnv, DoorKeyOptionalEnv]
            self._env_kwargs = [key_kwargs, gifts_kwargs, doorkeyoptional_kwargs]
        self.num_phases = len(self._envs)
        self._wrapper_seed = seed
        self._env_idx = None  # index of the current env
        self.env = KeyEnv(**key_kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

    def reset(self):
        """reset returns agent back to first environment, KeyEnv"""
        self._env_idx = 0
        self._wrapper_seed += 1
        self.env = self._envs[self._env_idx](
            **self._env_kwargs[self._env_idx],
            seed=self._wrapper_seed
        )
        observation = self.env.reset()
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done is True and self._env_idx < self.num_phases - 1:
            # to maintain compatibility with rendering
            self.env.render(close=True)
            if self._env_idx == 0:
                # if agent fetched key in first environment, initialize it with a key
                # in the last environment
                self._env_kwargs[-1]['key_color'] = info.get('carrying_key_color')
                if info.get('carrying_key_color') is not None:
                    print("Agent picked up key!")

            # teleport to the next environment
            self._env_idx += 1
            self.env = self._envs[self._env_idx](
                **self._env_kwargs[self._env_idx],
                seed=self._wrapper_seed
            )
            observation, done, info = self.env.reset(), False, {}

        return observation, reward, done, info

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def close(self):
        return self.env.close()

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)


class TinyKeyGiftsDoorEnv(KeyToGiftsToDoorKeyOptional):
    def __init__(self):
        key_kwargs = dict(
            size=6,
            key_color='yellow',
            start_by_key=False,
            max_steps=6**2
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
            max_steps=8**2
        )
        super().__init__(key_kwargs, gifts_kwargs, doorkeyoptional_kwargs)


class MediumKeyGiftsDoorEnv(KeyToGiftsToDoorKeyOptional):
    def __init__(self):
        key_kwargs = dict(
            size=6,
            key_color='yellow',
            start_by_key=False,
            max_steps=5*6**2
        )
        gifts_kwargs = dict(
            size=6,
            num_objs=2,
            gift_reward=0.1,
            max_steps=5*6**2
        )
        doorkeyoptional_kwargs = dict(
            size=6,
            key_color=None,
            door_color='yellow',
            max_steps=5*6**2
        )
        super().__init__(key_kwargs, gifts_kwargs, doorkeyoptional_kwargs)


class KeyNoGiftsDoorEnv(KeyToGiftsToDoorKeyOptional):
    def __init__(self):
        key_kwargs = dict(
            size=6,
            key_color='yellow',
            start_by_key=False,
            max_steps=5*6**2
        )
        gifts_kwargs = dict()
        doorkeyoptional_kwargs = dict(
            size=8,
            key_color=None,
            door_color='yellow',
            max_steps=5*8**2
        )
        super().__init__(key_kwargs, gifts_kwargs, doorkeyoptional_kwargs)


register(
    id='MiniGrid-KeyGiftsDoor-tiny-v0',
    entry_point='gym_minigrid.envs:TinyKeyGiftsDoorEnv'
)

register(
    id='MiniGrid-KeyGiftsDoor-medium-v0',
    entry_point='gym_minigrid.envs:MediumKeyGiftsDoorEnv'
)

register(
    id='MiniGrid-KeyNoGiftsDoor-v0',
    entry_point='gym_minigrid.envs:KeyNoGiftsDoorEnv'
)
