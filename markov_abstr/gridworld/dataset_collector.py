import argparse

import numpy as np
import torch
from tqdm import trange
from visgrid.gridworld import GridWorld
from visgrid.sensors import *


def get_dataset(n_samples: int, rows: int, cols: int, rearrange_xy: bool = False):
    env = GridWorld(rows=rows, cols=cols)
    sensor_list = []
    if rearrange_xy:
        sensor_list.append(RearrangeXYPositionsSensor((env._rows, env._cols)))
    sensor_list += [
        OffsetSensor(offset=(0.5, 0.5)),
        NoisySensor(sigma=0.05),
        ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3),
        # ResampleSensor(scale=2.0),
        BlurSensor(sigma=0.6, truncate=1.0),
        NoisySensor(sigma=0.01),
    ]
    sensor = SensorChain(sensor_list)

    s = env.get_state()
    states = [s]
    observations = [sensor.observe(s)]

    actions = []
    for t in trange(n_samples):
        a = np.random.choice(env.actions)
        s, _, _ = env.step(a)
        obs = sensor.observe(s)
        states.append(s)
        observations.append(obs)
        actions.append(a)
    states = np.stack(states)
    observations = np.stack(observations)

    s0 = torch.from_numpy(np.asarray(states[:-1, :]))
    s1 = torch.from_numpy(np.asarray(states[1:, :]))
    x0 = torch.from_numpy(np.asarray(observations[:-1, :]))
    x1 = torch.from_numpy(np.asarray(observations[1:, :]))
    c0 = s0[:, 0] * env._cols + s0[:, 1]
    a = torch.from_numpy(np.asarray(actions))

    torch.save(
        {
            "s0": s0,
            "s1": s1,
            "x0": x0,
            "x1": x1,
            "c0": c0,
            "a": a,
        },
        f'gridworld_{rows}x{cols}_{"rearranged" if rearrange_xy else "original"}.pt',
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--rows", type=int, default=6, help="Number of gridworld rows")
    parser.add_argument("-c", "--cols", type=int, default=6, help="Number of gridworld columns")
    parser.add_argument("--rearrange_xy", action="store_true", help="Rearrange discrete x-y positions to break smoothness")
    args = parser.parse_args()

    n_samples = 20000

    get_dataset(n_samples, args.rows, args.cols, args.rearrange_xy)
