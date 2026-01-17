import argparse
import mo_gymnasium as mo_gym
import numpy as np
import torch  # noqa: F401

from morl_baselines.multi_policy.morld.morld import MORLD


def parse_args():
    parser = argparse.ArgumentParser(description="Run MORL-D on MO-Gymnasium environments")
    
    # 1. Environment Name
    parser.add_argument(
        "--env_name", 
        type=str, 
        default="mo-halfcheetah-v5", 
        help="The Gymnasium environment ID"
    )
    
    # 2. Total Timesteps
    parser.add_argument(
        "--total_timesteps", 
        type=int, 
        default=int(3e6), 
        help="Total training timesteps"
    )
    
    # 3. Reference Point (accepts list of floats)
    parser.add_argument(
        "--ref_point", 
        type=float, 
        nargs="+", 
        default=[-100.0, -100.0],
        help="Reference point for hypervolume (space separated, e.g. -100 -100)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Convert ref_point list to numpy array
    ref_point = np.array(args.ref_point)
    gamma = 0.99

    print(f"Training on: {args.env_name}")
    print(f"Timesteps: {args.total_timesteps}")
    print(f"Reference Point: {ref_point}")

    # Initialize environment using the CLI argument
    env = mo_gym.make(args.env_name)
    eval_env = mo_gym.make(args.env_name)

    algo = MORLD(
        env=env,
        env_name=env.unwrapped.spec.id, # Often useful for wandb project naming
        exchange_every=int(5e4),
        pop_size=6,
        policy_name="MOSAC",
        scalarization_method="ws",
        evaluation_mode="ser",
        gamma=gamma,
        log=True,
        neighborhood_size=1,
        update_passes=10,
        shared_buffer=True,
        sharing_mechanism=[],
        weight_adaptation_method="PSA",
        seed=0,
    )

    algo.train(
        eval_env=eval_env,
        eval_ep_len=750,
        total_timesteps=args.total_timesteps + 1, # Using CLI argument
        ref_point=ref_point,                      # Using CLI argument
        known_pareto_front=None,
    )


if __name__ == "__main__":
    main()