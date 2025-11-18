"""
Quantum Circuit Compiler Training Script

This script trains a PPO reinforcement learning agent to optimize quantum circuits
by learning to apply transpiler passes in optimal sequences. The agent's goal is
to minimize circuit depth and gate count through intelligent pass selection.

Requirements:
- stable-baselines3: RL algorithms and utilities
- mlflow: Experiment tracking and model management
- qiskit: Quantum circuit manipulation
- gymnasium: RL environment interface

Usage:
1. Start MLflow UI: `mlflow ui` (optional, for monitoring)
2. Run training: `python train.py`
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import mlflow
import mlflow.pytorch
import os

from rl_compiler_env import QuantumCircuitEnv

# --- MLflow Configuration ---
# MLflow is used for experiment tracking,
# logging hyperparameters, metrics, and models
# To run MLflow UI: `mlflow ui` in terminal, then visit http://localhost:5000
# This allows you to compare different training runs and monitor progress
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
mlflow.set_experiment("Quantum_Compiler_PPO")  # Exp name for grouping runs


# --- Agent Evaluation Function ---
def evaluate_agent(model, env, n_episodes=100):
    """
    Evaluate the trained agent's performance on fresh quantum circuits.
    
    This function tests how well the agent can optimize new, unseen circuits
    by measuring the depth reduction achieved over multiple episodes.
    
    Args:
        model: Trained PPO model
        env: Quantum circuit environment (wrapped in DummyVecEnv)
        n_episodes: Number of test episodes to run
        
    Returns:
        tuple: (mean_reduction, std_reduction) - performance statistics
    """
    print("\nStarting evaluation...")
    all_reductions = []  # Store depth reduction for each episode

    for episode in range(n_episodes):
        # Reset environment to get a new random quantum circuit
        obs, info = env.reset()
        initial_depth = info['initial_depth']  # Record starting circuit depth

        terminated = False  # Episode completed (max steps reached)
        truncated = False   # Episode truncated (not used in this env)

        # Let the agent optimize the circuit for up to max_steps
        while not terminated and not truncated:
            # Use deterministic policy (no exploration) for evaluation
            action, _states = model.predict(obs, deterministic=True)
            # Apply the chosen transpiler pass and get new state
            obs, reward, terminated, truncated, info = env.step(action)

        final_depth = info['depth']  # Get final optimized circuit depth

        # Calculate percentage depth reduction achieved
        if initial_depth > 0:
            # Reduction = (original - final) / original
            # Positive values indicate improvement (depth reduced)
            reduction = (initial_depth - final_depth) / initial_depth
            all_reductions.append(reduction)
        else:
            # Edge case: circuit already had zero depth
            all_reductions.append(0)  # No change possible

        if (episode + 1) % 20 == 0:
            print(f"Evaluation episode {episode + 1}/{n_episodes} complete.")

    mean_reduction = np.mean(all_reductions)
    std_reduction = np.std(all_reductions)

    print("\n--- Evaluation Results ---")
    print(f"Episodes:         {n_episodes}")
    print(f"Mean Depth Reduction: {mean_reduction * 100:.2f}%")
    print(f"Std Dev Reduction:    {std_reduction * 100:.2f}%")
    print("----------------------------")

    return mean_reduction, std_reduction


# --- Main Training Script ---
def main():
    """Main training function that orchestrates the entire RL training process."""
    
    # Training configuration
    TIMESTEPS = 1_000_000  # Total training steps (1 M for good convergence)
    MODEL_PATH = "ppo_quantum_compiler.zip"  # Where to save the trained model

    # 1. Create and wrap the environment
    # DummyVecEnv is required by Stable Baselines3 for single environments
    # It provides vectorized interface even with just one environment
    env = DummyVecEnv([lambda: QuantumCircuitEnv(max_steps=20)])

    # 2. Define the PPO model with hyperparameters
    # MlpPolicy = Multi-Layer Perceptron (standard feed-forward neural network)
    model = PPO(
        "MlpPolicy",                              # Policy network architecture
        env,                                      # Environment to train on
        verbose=1,                                # Print training progress
        tensorboard_log="./ppo_tensorboard_log/",  # Log for TensorBoard vis
        
        # PPO-specific hyperparameters (tuned for this problem)
        n_steps=2048,        # Steps to collect before each policy update
        batch_size=64,       # Mini-batch size for gradient updates
        n_epochs=10,         # Number of epochs to train on collected data
        gamma=0.99,          # Discount factor (how much to value future rewards)
        gae_lambda=0.95,     # GAE parameter for advantage estimation
        clip_range=0.2,      # PPO clipping parameter (prevents large policy changes)
        ent_coef=0.01,       # Entropy coefficient (encourages exploration)
        learning_rate=3e-4,  # Learning rate for neural network updates
    )

    # 3. Start an MLflow run for experiment tracking
    with mlflow.start_run() as run:
        print(f"MLflow run started: {run.info.run_id}")

        # Log all hyperparameters for reproducibility and comparison
        # This allows you to track which settings led to best performance
        mlflow.log_params({
            "total_timesteps": TIMESTEPS,                    # Training duration
            "policy": "MlpPolicy",                          # Network architecture
            "env_max_steps": env.envs[0].max_steps,         # Episode length
            "n_steps": model.n_steps,                       # PPO rollout length
            "batch_size": model.batch_size,                 # Training batch size
            "n_epochs": model.n_epochs,                     # Epochs per update
            "gamma": model.gamma,                           # Discount factor
            "gae_lambda": model.gae_lambda,                 # GAE parameter
            "clip_range": model.clip_range,
            "ent_coef": model.ent_coef,
            "learning_rate": "3e-4",
        })

        # 4. Set up training with periodic evaluation
        # Create separate environment for evaluation (avoids interference with training)
        eval_env = DummyVecEnv([lambda: QuantumCircuitEnv(max_steps=20)])
        
        # EvalCallback monitors training progress and saves best models
        eval_callback = EvalCallback(
            eval_env,                              # Environment for evaluation
            best_model_save_path='./best_model/',  # Save best performing model
            log_path='./logs/',                    # Evaluation logs
            eval_freq=25000,                       # Evaluate every 25k steps
            deterministic=True,                    # Use deterministic policy for eval
            render=False                           # Don't render during evaluation
        )

        print("\nStarting model training...")
        # Main training loop - this is where the agent learns
        model.learn(
            total_timesteps=TIMESTEPS,  # Train for 1M steps
            callback=eval_callback,     # Periodic evaluation during training
            progress_bar=True          # Show progress bar
        )
        print("Training complete.")

        # 5. Save the final trained model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        # 6. Final evaluation on 100 test episodes
        # This gives us the final performance metrics
        mean_reduction, std_reduction = evaluate_agent(
            model, env, n_episodes=100
        )

        # 7. Log final performance metrics to MLflow
        # These metrics will appear in the MLflow UI for comparison
        mlflow.log_metrics({
            "mean_depth_reduction": mean_reduction,  # Average optimization performance
            "std_depth_reduction": std_reduction     # Consistency of performance
        })

        # Save the trained model as an MLflow artifact
        mlflow.log_artifact(MODEL_PATH)

        print(f"Run {run.info.run_id} finished. Check MLflow UI.")


if __name__ == '__main__':
    main()
