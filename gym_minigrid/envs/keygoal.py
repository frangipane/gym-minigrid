from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class KeyGoalEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a key and go to a goal.
    Agent does not receive immediate reward for picking up key, but gets
    it upon reaching the goal.
    """

    def __init__(self,
                 size=8,
                 key_color='yellow',
                 key_reward=3,
                 max_steps=2*8**2,
                 seed=1337):
        self.key_color = key_color
        self.key_reward = key_reward

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            seed=seed,
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

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Put key in a random location
        self.place_obj(Key(self.key_color))

        self.mission = "fetch the key and go to goal"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.carrying and self.carrying.type == 'key' and done is True:
            # delayed reward if carrying key at goal
            reward += self.key_reward

        return obs, reward, done, info


class KeyGoalEnv6x6(KeyGoalEnv):
    def __init__(self):
        super().__init__(size=6)


class KeyGoalEnv8x8(KeyGoalEnv):
    def __init__(self):
        super().__init__(size=8)


class KeyGoalEnv10x10(KeyGoalEnv):
    def __init__(self):
        super().__init__(size=10)


register(
    id='MiniGrid-KeyGoal-6x6-v0',
    entry_point='gym_minigrid.envs:KeyGoalEnv6x6'
)

register(
    id='MiniGrid-KeyGoal-8x8-v0',
    entry_point='gym_minigrid.envs:KeyGoalEnv8x8'
)

register(
    id='MiniGrid-KeyGoal-10x10-v0',
    entry_point='gym_minigrid.envs:KeyGoalEnv10x10'
)
