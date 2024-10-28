import random
import argparse
import json
import os
from typing import Union

#!! do not import matplotlib until you check input arguments
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from agents.a2cagent import DiscreteA2CAgent
from agents.reinforceagent import DiscreteReinforceAgent
from agents.dqnagent import DQNAgent
from agents.randomagent import RandomAgent
from models.nullabstraction import NullAbstraction
from models.phinet import PhiNet
from tqdm import tqdm
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

GAMMA: float = 0.9


def parse_args() -> argparse.Namespace:
    """Load and parse the CLI arguments for training the Markov agent."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
    # yapf: disable
    parser.add_argument("-a","--agent", type=str, required=True,
                        choices=["random","dqn", "reinforce", "a2c"], help="Type of agent to train")
    parser.add_argument("-n","--n_trials", type=int, default=1,
                        help="Number of trials")
    parser.add_argument("-e","--n_episodes", type=int, default=100,
                        help="Number of episodes per trial")
    parser.add_argument("-m","--max_steps", type=int, default=1000,
                        help="Maximum number of steps per episode")
    parser.add_argument("-r","--rows", type=int, default=6,
                        help="Number of gridworld rows")
    parser.add_argument("-c","--cols", type=int, default=6,
                        help="Number of gridworld columns")
    parser.add_argument("-w", "--walls", type=str, default="empty", choices=["empty", "maze", "spiral", "loop"],
                        help="The wall configuration mode of gridworld")
    parser.add_argument("-b","--batch_size", type=int, default=16,
                        help="Number of experiences to sample per batch")
    parser.add_argument("-l","--latent_dims", type=int, default=2,
                        help="Number of latent dimensions to use for representation")
    parser.add_argument("-lr","--learning_rate", type=float, default=0.001,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--use_gae", action="store_true",
                        help="Toggle between GAE and n-step returns in A2C")
    parser.add_argument("--n_steps", type=int, default=5,
                        help="Number of steps for n-step returns in A2C")
    parser.add_argument("--gae_lambda", type=float, default=1.0,
                        help="Lambda hyperparameter for generalized advantage estimation in A2C")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Entropy regularization for REINFORCE/A2C")
    parser.add_argument("--center_returns", type=bool, default=True,
                        help="Return centering for REINFORCE")
    parser.add_argument("-s","--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("-t","--tag", type=str, required=True,
                        help="Tag for identifying experiment")
    parser.add_argument("--phi_path", type=str,
                        help="Load an existing abstraction network by tag")
    parser.add_argument("--no_phi", action="store_true",
                        help="Turn off abstraction and just use observed state; i.e. ϕ(x)=x")
    parser.add_argument("--train_phi", action="store_true",
                        help="Allow simultaneous training of abstraction")
    parser.add_argument("--no_sigma", action="store_true",
                        help="Turn off sensors and just use true state; i.e. x=s")
    parser.add_argument("--one_hot", action="store_true",
                        help="Bypass sensor and use one-hot representation instead")
    parser.add_argument("--save", action="store_true",
                        help="Save final network weights")
    parser.add_argument("-v","--video", action="store_true",
                        help="Show video of agent training")
    parser.add_argument("--xy_noise", action="store_true",
                        help="Add truncated gaussian noise to x-y positions")
    parser.add_argument("--rearrange_xy", action="store_true",
                        help="Rearrange discrete x-y positions to break smoothness")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    # yapf: enable
    args = parser.parse_args()
    if args.train_phi and args.no_phi:
        assert False, "--no_phi and --train_phi are mutually exclusive"

    if args.one_hot and args.no_sigma:
        assert False, "--one_hot and --no_sigma are mutually exclusive"

    if args.video:
        import matplotlib.pyplot as plt
    return args


def load_environment(args: argparse.Namespace):
    log_dir = "results/scores/" + str(args.tag)
    os.makedirs(log_dir, exist_ok=True)
    log = open(log_dir + "/scores-{}-{}.txt".format(args.agent, args.seed), "w")

    if args.walls == "maze":
        env = MazeWorld.load_maze(rows=args.rows, cols=args.cols, seed=args.seed)
    elif args.walls == "spiral":
        env = SpiralWorld(rows=args.rows, cols=args.cols)
    elif args.walls == "loop":
        env = LoopWorld(rows=args.rows, cols=args.cols)
    else:
        env = GridWorld(rows=args.rows, cols=args.cols)

    sensor_list = []
    if args.xy_noise:
        sensor_list.append(NoisySensor(sigma=0.2, truncation=0.4))
    if args.rearrange_xy:
        sensor_list.append(RearrangeXYPositionsSensor((env._rows, env._cols)))
    if not args.no_sigma:
        if args.one_hot:
            sensor_list += [
                OffsetSensor(offset=(0.5, 0.5)),
                ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=1),
            ]
        else:
            sensor_list += [
                OffsetSensor(offset=(0.5, 0.5)),
                NoisySensor(sigma=0.05),
                ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3),
                # ResampleSensor(scale=2.0),
                BlurSensor(sigma=0.6, truncate=1.0),
                NoisySensor(sigma=0.01),
            ]
    sensor = SensorChain(sensor_list)
    return log, env, sensor


def load_agent(args: argparse.Namespace, sensor: SensorChain, env: GridWorld) -> Union[RandomAgent, DQNAgent]:
    """Initialize the agent."""

    if args.no_phi:
        phinet = NullAbstraction(-1, args.latent_dims)
    else:
        x0 = sensor.observe(env.get_state())
        phinet = PhiNet(input_shape=x0.shape, n_latent_dims=args.latent_dims, n_hidden_layers=1, n_units_per_layer=32)
        if args.phi_path:
            modelfile = f"results/models/{args.phi_path}/phi-{0}_latest.pytorch"
            phinet.load(modelfile)
    n_actions = 4
    if args.agent == "random":
        agent = RandomAgent(n_actions=n_actions)
    elif args.agent == "dqn":
        agent = DQNAgent(
            n_features=args.latent_dims,
            n_actions=n_actions,
            phi=phinet,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            train_phi=args.train_phi,
            gamma=GAMMA,
            factored=False,
        )
    elif args.agent == "reinforce":
        agent = DiscreteReinforceAgent(
            n_features=args.latent_dims,
            n_actions=n_actions,
            phi=phinet,
            lr=args.learning_rate,
            train_phi=args.train_phi,
            gamma=GAMMA,
            alpha=args.alpha,
            center_returns=args.center_returns,
        )
    elif args.agent == "a2c":
        agent = DiscreteA2CAgent(
            n_features=args.latent_dims,
            n_actions=n_actions,
            phi=phinet,
            lr=args.learning_rate,
            train_phi=args.train_phi,
            gamma=GAMMA,
            alpha=args.alpha,
            use_gae=args.use_gae,
            gae_lambda=args.gae_lambda,
            n_steps=args.n_steps,
        )

    else:
        assert False, f"Invalid agent type: {args.agent}"

    return agent


def main(args: argparse.Namespace) -> None:

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log, env, sensor = load_environment(args)
    agent = load_agent(args, sensor, env)

    if args.video:
        fig, ax = plt.subplots(2, 2)
        ax = ax.flatten()
        fig.show()

        def plot_value_function(ax):
            s = np.asarray([[np.asarray([x, y]) for x in range(args.cols)] for y in range(args.rows)])
            v = np.asarray(agent.q_values(s).detach().numpy()).max(-1)
            xy = OffsetSensor(offset=(0.5, 0.5)).observe(s).reshape(args.cols, args.rows, -1)
            ax.contourf(np.arange(0.5, args.cols + 0.5), np.arange(0.5, args.rows + 0.5), v, vmin=-10, vmax=0)

        def plot_states(ax):
            data = pd.DataFrame(agent.replay.memory)
            data[["x.r", "x.c"]] = pd.DataFrame(data["x"].tolist(), index=data.index)
            data[["xp.r", "xp.c"]] = pd.DataFrame(data["xp"].tolist(), index=data.index)
            sns.scatterplot(
                data=data,
                x="x.c",
                y="x.r",
                hue="done",
                style="done",
                markers=True,
                size="done",
                size_order=[1, 0],
                ax=ax,
                alpha=0.3,
                legend=False,
            )
            ax.invert_yaxis()

    wandb.init(
        project="grid_representations", name=args.agent, config=vars(args), mode="online" if args.wandb else "disabled"
    )

    for trial in tqdm(range(args.n_trials), desc="trials"):
        # Set a random position for the goal
        env.reset_goal()
        agent.reset()
        total_reward = 0
        total_steps = 0
        losses = []
        rewards = []
        value_fn = []
        for episode in tqdm(range(args.n_episodes), desc="episodes"):
            # Reset the agent to a random position
            env.reset_agent()
            ep_rewards = []
            for step in range(args.max_steps):
                s = env.get_state()
                x = sensor.observe(s)

                a = agent.act(x)
                sp, r, done = env.step(a)
                xp = sensor.observe(sp)
                ep_rewards.append(r)
                if args.video and isinstance(agent, DQNAgent):
                    value_fn.append(agent.v(x))
                total_reward += r

                if isinstance(agent, DiscreteReinforceAgent):
                    agent.replay.add_experience(x, a, r, done)
                elif isinstance(agent, DiscreteA2CAgent):
                    agent.buffer.add_experience(x, a, r, xp, done)
                    if step > 0 and step % agent.n_steps == 0:
                        loss = agent.train()
                else:
                    loss = agent.train(x, a, r, xp, done)
                    losses.append(loss)
                rewards.append(r)

                if done:
                    if isinstance(agent, DiscreteReinforceAgent):
                        loss = agent.train()
                        losses.append(loss)
                    break

            if args.video:
                [a.clear() for a in ax]
                plot_value_function(ax[0])
                env.plot(ax[0])
                ax[1].plot(value_fn)
                ax[2].plot(rewards, c="C3")
                ax[3].plot(losses, c="C1")
                # plot_states(ax[3])
                ax[1].set_ylim([-10, 0])
                fig.canvas.draw()
                fig.canvas.flush_events()

            total_steps += step
            score_info = {
                "trial": trial,
                "episode": episode,
                "reward": sum(ep_rewards),
                "total_reward": total_reward,
                "total_steps": total_steps,
                "steps": step,
            }
            wandb.log(score_info)
            json_str = json.dumps(score_info)
            log.write(json_str + "\n")
            log.flush()
    print("\n\n")

    if args.save:
        agent.q.save("qnet-{}".format(args.seed), "results/models/{}".format(args.tag))


if __name__ == "__main__":
    args = parse_args()
    main(args)
