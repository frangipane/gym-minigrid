from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class KeyEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a key.  
    Episode ends when key is fetched, or timeout, but no reward is given in
    either case.
    """

    def __init__(self, size=8, key_color='yellow', start_by_key=False):
        self.key_color = key_color
        self._start_by_key = start_by_key

        super().__init__(
            grid_size=size,
            max_steps=2*size**2,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        if self._start_by_key is True:
            # put key in front of agent, so agent has 1/num_actions
            # probability of picking up key in its first action.
            self.put_obj(Key(self.key_color),
                         self.agent_pos[0]+1,
                         self.agent_pos[1])
        else:
            # Put key in bottom-right corner
            self.put_obj(Key(self.key_color), width-2, height-2)

        self.mission = "fetch a key"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.carrying and self.carrying.type == 'key':
            info['carrying_key_color'] = self.carrying.color
            reward = 0
            done = True

        return obs, reward, done, info


class KeyEnv8x8(KeyEnv):
    def __init__(self):
        super().__init__(size=8)


class KeyEnv8x8StartByKey(KeyEnv):
    def __init__(self):
        super().__init__(size=8, start_by_key=True)


class KeyEnv10x10(KeyEnv):
    def __init__(self):
        super().__init__(size=10)


register(
    id='MiniGrid-Key-8x8-v0',
    entry_point='gym_minigrid.envs:KeyEnv8x8'
)

register(
    id='MiniGrid-Key-8x8-startbykey-v0',
    entry_point='gym_minigrid.envs:KeyEnv8x8StartByKey'
)

register(
    id='MiniGrid-Key-10x10-v0',
    entry_point='gym_minigrid.envs:KeyEnv10x10'
)
