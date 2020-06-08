from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class GoalKeyOptionalEnv(MiniGridEnv):
    """
    Environment in which the agent has to reach the goal.  If agent is initialized
    already carrying a key (of any color), agent is awarded extra upon reaching the goal.
    """

    def __init__(self,
                 size=8,
                 carrying=None,
                 key_reward=3,
                 max_steps=8**2,
                 seed=1337,
                 goal_reward=1.,
    ):
        self._carrying = carrying
        self.key_reward = key_reward
        self.goal_reward = goal_reward

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            seed=seed,
        )

    def reset(self):
        """Override reset so that agent can be initialized
        carrying a key already"""

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        self._gen_grid(self.width, self.height)

        # Item picked up, being carried
        self.carrying = self._carrying

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

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

        self.mission = "go to goal"

    def _reward(self):
        """
        Compute the reward to be given upon reaching goal.  Overrides default
        of 1 - 0.9 * (self.step_count / self.max_steps)
        """
        reward = self.goal_reward

        if self.carrying and self.carrying.type == 'key':
            # extra reward if carrying key at goal
            reward += self.key_reward

        return reward - 0.9 * (self.step_count / self.max_steps)


class GoalKeyOptionalEnvNoKey6x6(GoalKeyOptionalEnv):
    def __init__(self):
        super().__init__(size=6, max_steps=10*6**2, carrying=None)


class GoalKeyOptionalEnvWithKey6x6(GoalKeyOptionalEnv):
    def __init__(self):
        super().__init__(size=6, max_steps=10*6**2, carrying=Key('yellow'))


register(
    id='MiniGrid-GoalKeyOptionalEnvNoKey-6x6-v0',
    entry_point='gym_minigrid.envs:GoalKeyOptionalEnvNoKey6x6'
)


register(
    id='MiniGrid-GoalKeyOptionalEnvWithKey-6x6-v0',
    entry_point='gym_minigrid.envs:GoalKeyOptionalEnvWithKey6x6'
)
