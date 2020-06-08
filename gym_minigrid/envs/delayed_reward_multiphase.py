import warnings

import gym
from gym_minigrid.register import register
from gym_minigrid.envs.fetchkey import KeyEnv


class ThreePhaseDelayedReward(gym.Env):
    """
    Stitches together three environments: 
    (1) KeyEnv -- an environment where agent is instantiated in an empty grid with
        a randomly placed key (agent does not receive immediate reward for picking up
        the key, but can received delayed reward in the 3rd phase depending of if it
        picked up the key in the first phase.
    (2) A distractor env, e.g. GiftsEnv -- this phase is optional (by setting max_steps to 0)
    (3) An env that gives a delayed reward depending on whether key was picked up in
        the first phase in KeyEnv, e.g. GoalKeyOptionalEnv and DoorKeyOptionalEnv

    Agent is teleported from (1) to (2) to (3).  The agent is initialized with or
    without the key in (2) and (3), depending on whether it successfully fetched it
    in (1) and did not drop it before phase (3).

    Args
    ----
    key_kwargs (dict) : Any kwargs for the KeyEnv environment.

    distractor_env : A function which creates a copy of the environment.

    distractor_kwargs (dict) : Any kwargs for the distractor environment.
        The distractor_env constructor must take `carrying` (None or WorldObject)
        as a kwarg.

    delayed_reward_env : A function which creates a copy of the environment.
        The delayed_reward_env constructor must take `carrying` (None or string)
        as a kwarg.

    delayed_reward_kwargs (dict) : Any kwargs for the distractor environment.
    
    seed (int) : Seed for random number generators.   
    """
    def __init__(self,
                 key_kwargs,
                 distractor_env,
                 distractor_kwargs,
                 delayed_reward_env,
                 delayed_reward_kwargs,
                 seed=111):
        self._envs = [KeyEnv, distractor_env, delayed_reward_env]
        self._env_kwargs = [key_kwargs, distractor_kwargs, delayed_reward_kwargs]
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
        self.env = self._envs[0](**self._env_kwargs[0], seed=self._wrapper_seed)
        observation = self.env.reset()
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done is True and self._env_idx < self.num_phases - 1:
            # to maintain compatibility with rendering
            self.env.render(close=True)
            if self._env_idx == 0 and self.carrying and self.carrying.type == 'key':
                print("Agent picked up key!")

            # If agent finished the current phase while carrying an object,
            # then initialize it in next phase carrying the same object
            self._env_kwargs[self._env_idx + 1]['carrying'] = self.carrying

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
        self._wrapper_seed = seed
        return self.env.seed(seed)

    def close(self):
        return self.env.close()

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)


#==============================================================================================
# Register some different combinations of distractor and delayed reward phases


from gym_minigrid.envs.opengifts import GiftsEnv
from gym_minigrid.envs.doorkeyoptional import DoorKeyOptionalEnv
from gym_minigrid.envs.goalkeyoptional import GoalKeyOptionalEnv


class TinyKeyGiftsDoorEnv(ThreePhaseDelayedReward):
    def __init__(self):
        super().__init__(
            key_kwargs=dict(
                size=6,
                key_color='yellow',
                start_by_key=False,
                max_steps=5*6**2
            ),
            distractor_env=GiftsEnv,
            distractor_kwargs=dict(
                size=6,
                num_objs=3,
                gift_reward=1,
                max_steps=5*6**2
            ),
            delayed_reward_env=DoorKeyOptionalEnv,
            delayed_reward_kwargs=dict(
                size=8,
                key_color=None,
                door_color='yellow',
                max_steps=5*8**2
            )
        )


class KeyNoDistractorDoorEnv(ThreePhaseDelayedReward):
    def __init__(self):
        super().__init__(
            key_kwargs = dict(
                size=6,
                key_color='yellow',
                start_by_key=False,
                max_steps=5*6**2
            ),
            distractor_env=GiftsEnv,
            distractor_kwargs = dict(
                size=6,
                num_objs=2,
                gift_reward=0.1,
                max_steps=0,
            ),
            delayed_reward_env=DoorKeyOptionalEnv,
            delayed_reward_kwargs = dict(
                size=8,
                key_color=None,
                door_color='yellow',
                max_steps=5*8**2
            )
        )


class TinyKeyGiftsGoalEnv(ThreePhaseDelayedReward):
    def __init__(self):
        super().__init__(
            key_kwargs=dict(
                size=6,
                key_color='yellow',
                start_by_key=False,
                max_steps=5*6**2,
                done_when_fetched=False,
            ),
            distractor_env=GiftsEnv,
            distractor_kwargs=dict(
                size=6,
                num_objs=3,
                gift_reward=1,
                max_steps=5*6**2,
                done_when_all_opened=False,
            ),
            delayed_reward_env=GoalKeyOptionalEnv,
            delayed_reward_kwargs=dict(
                size=8,
                carrying=None,
                max_steps=5*8**2,
                goal_reward=1.,
                key_reward=4.,
            )
        )


register(
    id='MiniGrid-KeyGiftsDoor-tiny-v0',
    entry_point='gym_minigrid.envs:TinyKeyGiftsDoorEnv'
)


register(
    id='MiniGrid-KeyNoDistractorDoor-v0',
    entry_point='gym_minigrid.envs:KeyNoDistractorDoorEnv'
)


register(
    id='MiniGrid-KeyGiftsGoal-tiny-v0',
    entry_point='gym_minigrid.envs:TinyKeyGiftsGoalEnv'
)
