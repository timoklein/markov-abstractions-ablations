
from typing import Any, Dict, Tuple, List

import numpy as np

from visgrid.gridworld import (
    GridWorld,
    LoopWorld,
    MazeWorld,
    RingWorld,
    SnakeWorld,
    SpiralWorld,
    TestWorld,
)
from visgrid.sensors import *
from train_agent import GAMMA

TransitionMatrix = Dict[int, Dict[int, List[Tuple[Any, ...]]]]

EPSILON = 1e-4  # Stopping criterion

CONFIG = {
    "rows": 6,
    "cols": 6,
    "walls": "empty",
}

def load_environment(cfg: Dict[str, Any]):

    if cfg["walls"] == "maze":
        env = MazeWorld.load_maze(rows=cfg["rows"], cols=cfg["cols"], seed=cfg["seed"])
    elif cfg["walls"] == "spiral":
        env = SpiralWorld(rows=cfg["rows"], cols=cfg["cols"])
    elif cfg["walls"] == "loop":
        env = LoopWorld(rows=cfg["rows"], cols=cfg["cols"])
    else:
        env = GridWorld(rows=cfg["rows"], cols=cfg["cols"])

    # sensor_list = []
    # if cfg["rearrange_xy"]:
    #     sensor_list.append(RearrangeXYPositionsSensor((env._rows, env._cols)))
    # sensor = SensorChain(sensor_list)
    return env

def _one_step_lookahead(state: int, transition_matrix: TransitionMatrix, Q_function: np.ndarray) -> np.ndarray:
    """Perform a 1-step lookahead and use the Bellman equation to calculate the expected value for all actions in a state."""
    dynamics = transition_matrix[state]

    action_vals = np.zeros(len(dynamics.keys()))
    # Since transitions are stochastic, we must loop over all transitions
    for action, action_transitions in dynamics.items():
        for transition in action_transitions:
            transition_prob, next_state, reward, is_terminal = transition
            next_x, next_y = next_state
            # Calculate the 1-step lookahead reward value for transition using the Bellman optimality equation
            transition_reward = transition_prob * (reward + (1 - is_terminal) * GAMMA * Q_function[next_x, next_y, :].max())
            action_vals[action] += transition_reward

    return action_vals


def q_value_iteration(env: GridWorld, Q_function: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs 1-step Bellman optimality backups until the stopping criterion is met."""

    # Loop until target accuracy is reached
    iteration = 0
    while True:
        max_diff = 0
        for state in env.state_list:
            x, y = state
            # Calculate new value Q for state s with Bellman optimality backup
            # P is the environment's stochastic transition matrix
            # It is indexed by each state from 0 to n-1
            # Indexing with a state returns a dictionary that contains 4 keys
            # Each key represents an action and has a list as value
            # The list contains arrays of the form (transition_prob, next_state, reward, is_terminal)
            # This is necessary because transitions are stochastic!
            action_vals = _one_step_lookahead(state, env.P, Q_function)

            diff = abs(Q_function[x, y, :] - action_vals).max()

            # Do an in-place update of all Q values for state s
            Q_function[x, y, :] = action_vals

            print(f"Successful update at {iteration}. Value of starting state: {Q_function[0, 0]}")
            iteration += 1  
            
            if max_diff < diff:
                max_diff = diff
        # Termine loop if desired accuracy is reached
        if max_diff < EPSILON:
            return Q_function


if __name__ == "__main__":
    env = load_environment(CONFIG)
    env.reset_goal()

    Q_function = np.zeros((env._rows, env._cols, len(env.actions)))

    Q_opt = q_value_iteration(env, Q_function)
    s = env.get_state()
    done = False
    import ipdb; ipdb.set_trace(context=21)
    while not done:
        x, y = s
        action = Q_opt[x, y, :].argmax()
        s, reward, done = env.step(action)