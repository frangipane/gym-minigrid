from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class DoorKeyOptionalEnv(MiniGridEnv):
    """
    Environment with a yellow door and no key, sparse reward.  Agent
    must unlock door to reach goal.

    Agent cannot solve task unless it is already carrying a key matching 
    the color of the door when the environment is initialized.
    """

    def __init__(self,
                 size=8,
                 carrying=None,
                 door_color='yellow',
                 door_reward=0.0,
                 max_steps=10*8**2,
                 seed=1337,
                 goal_reward=1,
    ):
        self._door_reward = door_reward
        self._carrying = carrying  # key must be same color as door to solve task
        self._door_color = door_color
        self._door_num_times_opened = 0
        self._goal_reward = goal_reward
        super().__init__(grid_size=size, max_steps=max_steps, seed=seed)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # agent opened the single door
        if action == self.actions.toggle and info.get('toggle_succeeded') is True:
            # Get the position in front of the agent
            fwd_pos = self.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Only reward first opening of door
            if fwd_cell.type == 'door' and self._door_num_times_opened == 0:
                reward += self._door_reward
                self._door_num_times_opened += 1

        return obs, reward, done, info

    def reset(self):
        """Override reset so that agent can be initialized
        carrying a key already"""

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

        # Item picked up, being carried
        self.carrying = self._carrying

        # Step count since episode start
        self.step_count = 0

        # Reset count of times door opened
        self._door_num_times_opened = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door(self._door_color, is_locked=True), splitIdx, doorIdx)

        self.mission = "use the key to open the door and then get to the goal"

    def _reward(self):
        """
        Compute the reward to be given upon success.  Overrides default
        of 1 - 0.9 * (self.step_count / self.max_steps)
        """
        return self._goal_reward - 0.9 * (self.step_count / self.max_steps)


class DoorHasKey8x8Env(DoorKeyOptionalEnv):
    def __init__(self):
        super().__init__(size=8, carrying=Key('yellow'), door_color='yellow')


class DoorNoKey8x8Env(DoorKeyOptionalEnv):
    def __init__(self):
        super().__init__(size=8, carrying=None)


register(
    id='MiniGrid-DoorHasKey-8x8-v0',
    entry_point='gym_minigrid.envs:DoorHasKey8x8Env'
)

register(
    id='MiniGrid-DoorNoKey-8x8-v0',
    entry_point='gym_minigrid.envs:DoorNoKey8x8Env'
)
