import numpy as np
from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class GiftsEnv(MiniGridEnv):
    """
    Environment in which the agent has to open randomly 
    placed gifts

    gift_reward : scalar, or list/tuple
       if list/tuple, then a range of [min, max) specifying the reward range
       to uniformly sample from.

    done_when_all_opened : bool, if True, env returns done as True
       when all gifts have been opened.  Otherwise, done only
       happens at timeout from reaching max_steps.
    """

    def __init__(self,
                 size=8,
                 num_objs=3,
                 gift_reward=10.,
                 max_steps=5*8**2,
                 done_when_all_opened=False,
                 seed=1337
    ):
        if not isinstance(gift_reward, (list, tuple)):
            self._gift_reward = [gift_reward, gift_reward]

        if num_objs < 1:
            raise ValueError(f"num_objs must be an integer greater than 0")
        self.num_objs = num_objs
        self._num_opened = 0
        self._done_when_all_opened = done_when_all_opened

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            seed=seed
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        placed_count = 0
        # For each object to be generated
        while placed_count < self.num_objs:
            self.place_obj(Gift())
            placed_count += 1

        # Randomize the player start position and orientation
        self.place_agent()
        self.mission = 'Open all the gifts'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # agent opened a gift
        if action == self.actions.toggle and info.get('toggle_succeeded') is True:
            # Get the position in front of the agent
            fwd_pos = self.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell.type == 'gift':
                reward += np.random.uniform(*self._gift_reward)
                self._num_opened += 1
                if self._done_when_all_opened and self._num_opened == self.num_objs:
                    done = True

        return obs, reward, done, info

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Number of opened gifts
        self._num_opened = 0

        # Return first observation
        obs = self.gen_obs()
        return obs


class GiftsEnv8x8N3Rew10(GiftsEnv):
    def __init__(self):
        super().__init__(size=8, num_objs=3, gift_reward=10)


class GiftsEnv15x15N15Rew10(GiftsEnv):
    def __init__(self):
        super().__init__(size=15, num_objs=15, gift_reward=10)


register(
    id='MiniGrid-Gifts-8x8-N3-Rew10-v0',
    entry_point='gym_minigrid.envs:GiftsEnv8x8N3Rew10'
)

register(
    id='MiniGrid-Gifts-15x15-N15-Rew10-v0',
    entry_point='gym_minigrid.envs:GiftsEnv15x15N15Rew10'
)
