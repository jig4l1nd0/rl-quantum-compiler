#!/usr/bin/env python3
"""
Enhanced RL training script with action diversity improvements.
This addresses the Action 3 bias by training on circuits designed for each action.
"""

import os
import json
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from rl_compiler_env import QuantumCircuitEnv
from qiskit import QuantumCircuit
from qiskit.qasm2 import loads as qasm2_loads, dumps as qasm2_dumps


class ActionBalanceCallback(BaseCallback):
    """Callback to monitor action usage during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_counts = {}
        
    def _on_step(self) -> bool:
        # Track actions taken by the agent
        try:
            if hasattr(self.training_env, 'get_attr'):
                actions = self.training_env.get_attr('last_action')
                for action in actions:
                    if action is not None:
                        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        except:
            pass  # Skip if can't access action info
        
        # Log action distribution every 10k steps
        if self.num_timesteps % 10000 == 0 and self.action_counts:
            print(f"\nAction distribution at step {self.num_timesteps}:")
            total_actions = sum(self.action_counts.values())
            for action_id in range(14):
                count = self.action_counts.get(action_id, 0)
                percentage = (count / total_actions * 100) if total_actions > 0 else 0
                print(f"  Action {action_id}: {count} ({percentage:.1f}%)")
        
        return True


class EnhancedQuantumCircuitEnv(QuantumCircuitEnv):
    """Enhanced environment with action diversity incentives."""
    
    def __init__(self, action_specific_circuits=None):
        super().__init__()
        self.action_specific_circuits = action_specific_circuits or {}
        self.action_history = []
        self.last_action = None
        self.episode_count = 0
        
    def reset(self, **kwargs):
        """Reset with possibility of using action-specific circuits."""
        self.action_history = []
        self.last_action = None
        self.episode_count += 1
        
        # 30% chance to use action-specific circuit for targeted training
        if random.random() < 0.3 and self.action_specific_circuits:
            action_type = random.choice(list(self.action_specific_circuits.keys()))
            circuit_qasm = random.choice(self.action_specific_circuits[action_type])
            try:
                target_circuit = qasm2_loads(circuit_qasm)
                # Set the circuit directly for this episode
                self._initialize_circuit(target_circuit)
                obs = self._get_obs()
                return obs, {}
            except:
                pass  # Fall back to random circuit if parsing fails
        
        # Default random circuit generation
        return super().reset(**kwargs)
    
    def step(self, action):
        """Enhanced step with action diversity rewards."""
        self.last_action = action
        self.action_history.append(action)
        
        # Get base step result
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Apply action diversity bonuses/penalties
        reward = self._apply_diversity_reward(reward, action)
        
        return obs, reward, terminated, truncated, info
    
    def _apply_diversity_reward(self, base_reward, action):
        """Apply reward modifications to encourage action diversity."""
        modified_reward = base_reward
        
        # Track recent actions (last 5 steps)
        recent_actions = self.action_history[-5:]
        
        # Diversity bonus: reward using different actions
        if len(recent_actions) >= 3:
            unique_recent = len(set(recent_actions))
            if unique_recent > 1:
                diversity_bonus = 0.2 * unique_recent  # Bonus for variety
                modified_reward += diversity_bonus
        
        # Repetition penalty: discourage repeated ineffective actions
        if len(recent_actions) >= 4:
            last_four = recent_actions[-4:]
            if len(set(last_four)) == 1 and base_reward < 0.5:
                # Same action 4 times with low reward = penalty
                repetition_penalty = -0.3
                modified_reward += repetition_penalty
        
        # Action 3 bias reduction: slightly reduce reward for Action 3
        # to prevent over-reliance (but still allow it when effective)
        if action == 3 and base_reward < 1.0:
            action3_reduction = -0.1
            modified_reward += action3_reduction
            
        return modified_reward
    
    def _initialize_circuit(self, circuit):
        """Initialize environment with a specific circuit."""
        self.circuit = circuit.copy()
        self.initial_depth = circuit.depth()
        self.initial_size = circuit.size()
        self.prev_depth = self.initial_depth
        self.prev_size = self.initial_size
        self.current_step = 0


def load_action_specific_circuits():
    """Load pre-generated action-specific circuits."""
    circuits_file = 'action_specific_training_circuits.json'
    
    if os.path.exists(circuits_file):
        with open(circuits_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: {circuits_file} not found. Run training_circuits_generator.py first.")
        print("Training will proceed with random circuits only.")
        return {}


def train_enhanced_model():
    """Train PPO model with action diversity enhancements."""
    
    print("🚀 Starting Enhanced RL Training with Action Diversity")
    print("=" * 60)
    
    # Load action-specific circuits
    action_circuits = load_action_specific_circuits()
    if action_circuits:
        total_circuits = sum(len(circuits) for circuits in action_circuits.values())
        print(f"📊 Loaded {total_circuits} action-specific training circuits")
        for action_type, circuits in action_circuits.items():
            print(f"  - {action_type}: {len(circuits)} circuits")
    else:
        print("⚠️  No action-specific circuits loaded - using random generation only")
    
    print(f"\n🎯 Training Configuration:")
    print(f"  - Total timesteps: 300,000 (3x original)")
    print(f"  - Learning rate: 0.0003")
    print(f"  - Action diversity incentives: Enabled")
    print(f"  - Circuit balancing: 30% action-specific, 70% random")
    
    # Create enhanced environment
    def make_env():
        return EnhancedQuantumCircuitEnv(action_specific_circuits=action_circuits)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Initialize PPO with modified hyperparameters for better exploration
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # Increased entropy for more exploration
        verbose=1,
        tensorboard_log="./enhanced_ppo_logs/",
        seed=42
    )
    
    # Add action balance monitoring
    action_callback = ActionBalanceCallback(verbose=1)
    
    print(f"\n🎮 Starting training...")
    print("Monitor action diversity in the logs above.")
    
    # Train the model
    model.learn(
        total_timesteps=300000,
        callback=[action_callback],
        progress_bar=True
    )
    
    # Save the enhanced model
    model.save("ppo_quantum_compiler_enhanced")
    print(f"\n✅ Enhanced model saved as 'ppo_quantum_compiler_enhanced.zip'")
    
    # Final action distribution report
    if action_callback.action_counts:
        print(f"\n📊 Final Action Distribution:")
        total = sum(action_callback.action_counts.values())
        for action_id in range(14):
            count = action_callback.action_counts.get(action_id, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  Action {action_id}: {count:5d} ({percentage:5.1f}%)")
    
    return model


def test_enhanced_model():
    """Test the enhanced model's action diversity."""
    print(f"\n🧪 Testing Enhanced Model Action Diversity")
    print("=" * 50)
    
    # Load the enhanced model
    try:
        env = EnhancedQuantumCircuitEnv()
        model = PPO.load("ppo_quantum_compiler_enhanced", env=env)
        print("✅ Enhanced model loaded successfully")
    except:
        print("❌ Enhanced model not found. Train it first with train_enhanced_model()")
        return
    
    # Test circuits designed for different actions
    test_circuits = {
        "CNOT Cancellation": """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
cx q[0], q[1];
cx q[0], q[1];
measure q -> c;""",
        "Gate Consolidation": """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
rx(0.1) q[0];
ry(0.2) q[0];
rz(0.3) q[0];
measure q -> c;""",
        "Barrier Removal": """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
barrier q;
cx q[0], q[1];
barrier q;
measure q -> c;"""
    }
    
    for circuit_name, qasm_code in test_circuits.items():
        print(f"\n🔬 Testing: {circuit_name}")
        try:
            circuit = qasm2_loads(qasm_code)
            env._initialize_circuit(circuit)
            obs = env._get_obs()
            
            actions_used = []
            for step in range(5):  # Test 5 steps
                action, _states = model.predict(obs, deterministic=False)  # Non-deterministic for exploration
                action = int(action)  # Convert numpy array to int
                actions_used.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"  Step {step+1}: Action {action}, Reward: {reward:.2f}")
                if terminated or truncated:
                    break
            
            unique_actions = len(set(actions_used))
            print(f"  📊 Actions used: {actions_used}")
            print(f"  🎯 Action diversity: {unique_actions}/5 unique actions")
            
        except Exception as e:
            print(f"  ❌ Error testing {circuit_name}: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_enhanced_model()
    else:
        # Generate training circuits first if they don't exist
        if not os.path.exists('action_specific_training_circuits.json'):
            print("📋 Generating action-specific training circuits...")
            os.system("python training_circuits_generator.py")
        
        # Train enhanced model
        train_enhanced_model()
        
        # Test the results
        print("\n" + "=" * 60)
        test_enhanced_model()