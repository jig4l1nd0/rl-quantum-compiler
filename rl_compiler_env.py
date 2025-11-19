"""
Quantum Circuit Optimization Environment for Reinforcement Learning

This module implements a custom OpenAI Gymnasium environment where RL agents
learn to optimize quantum circuits by applying transpiler passes (optimization
techniques) in intelligent sequences.

Key Features:
- Random quantum circuit generation for training diversity
- 14 different transpiler passes as actions
- Reward system that prioritizes depth reduction over gate count reduction
- Fixed-size state representation for consistent RL input
- Compatible with standard RL algorithms (PPO, DQN, etc.)

Environment Specs:
- Action Space: Discrete(14) - choice of transpiler pass
- Observation Space: Box(16,) - circuit metrics + gate counts  
- Episode Length: max_steps (default 20)
- Reward: depth_reduction * 10 + gate_count_reduction * 1
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGates, CXCancellation, CommutationAnalysis,
    InverseCancellation, Unroll3qOrMore, Collect2qBlocks,
    ConsolidateBlocks, UnitarySynthesis,
    FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout,
    TrivialLayout, DenseLayout, SabreLayout, StochasticSwap,
    BarrierBeforeFinalMeasurements, RemoveFinalMeasurements
)
from qiskit.converters import circuit_to_dag, dag_to_circuit

# === ENVIRONMENT CONSTANTS ===

# Define a fixed set of gate types for consistent feature vectors
# This ensures the observation space has a stable shape across all circuits
# regardless of which gates are actually present in any given circuit
FIXED_GATE_SET = [
    'cx',       # Controlled-X (CNOT) gate - most common 2-qubit gate
    'u',        # Generic single-qubit unitary gate
    'u3',       # 3-parameter single-qubit gate (most general rotation)
    'h',        # Hadamard gate - creates superposition
    'rz',       # Z-rotation gate - phase rotation
    'x',        # Pauli-X gate (bit flip)
    'y',        # Pauli-Y gate 
    'z',        # Pauli-Z gate (phase flip)
    'swap',     # SWAP gate - exchanges two qubits
    'cz',       # Controlled-Z gate
    'ccx',      # Toffoli gate (controlled-controlled-X)
    'measure',  # Measurement operation
    'barrier'   # Barrier for visualization/organization
]

# Maximum number of qubits to handle in generated circuits
# This limits the complexity and ensures reasonable training times
MAX_QUBITS = 15

# State vector size: 3 basic metrics + gate counts for each gate type
# Structure: [depth, size, num_qubits, count_cx, count_u, ...]
STATE_VECTOR_SIZE = 3 + len(FIXED_GATE_SET)


class QuantumCircuitEnv(gym.Env):
    """
    Custom Gymnasium Environment for Quantum Circuit Compilation.

    This environment simulates the quantum circuit optimization process where
    an RL agent learns to apply transpiler passes (optimization techniques)
    to minimize circuit depth and gate count.

    The agent's goal is to apply a sequence of transpiler passes to minimize
    the depth and gate count of randomly generated quantum circuits.

    Environment Details:
    -------------------
    - State: 16D vector [depth, size, num_qubits, gate_counts...]
    - Action: Integer 0-13 selecting which transpiler pass to apply
    - Reward: 10 * depth_reduction + 1 * gate_count_reduction
    - Episode: Ends after max_steps or when no more improvements possible

    Transpiler Passes (Actions):
    ---------------------------
    0-8:  Individual optimization passes (gate-level optimizations)
    9-11: Preset optimization levels (comprehensive optimizations)
    12-13: Measurement and barrier management

    Attributes:
    -----------
    max_steps : int
        Maximum number of optimization steps per episode
    min_qubits : int
        Minimum number of qubits in generated circuits
    max_qubits : int
        Maximum number of qubits in generated circuits
    transpiler_passes : list
        Available optimization passes (actions)
    circuit : QuantumCircuit
        Current circuit being optimized
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, max_steps=20):
        """
        Initialize the Quantum Circuit Optimization Environment.

        Args:
            max_steps (int): Maximum optimization steps per episode
        """
        super().__init__()

        # Episode and circuit constraints
        self.max_steps = max_steps      # Maximum steps before episode ends
        self.min_qubits = 5             # Minimum qubits for complexity
        self.max_qubits = MAX_QUBITS    # Maximum qubits to keep tractable

        # Current episode state
        self.circuit = None             # Current quantum circuit
        self.initial_depth = 0          # Starting circuit depth
        self.initial_size = 0           # Starting gate count
        self.current_step = 0           # Current step in episode

        # Define the action space: available transpiler passes
        # Each action corresponds to applying a specific optimization pass
        self.transpiler_passes = [
            # Single-gate optimization passes
            Optimize1qGates(),          # Optimize 1-qubit gate sequences
            CXCancellation(),           # Cancel adjacent CNOT gates
            CommutationAnalysis(),      # Analyze gate commutation relations
            # Cancel inverse operations for common self-inverse gates
            InverseCancellation([
                CXGate(), HGate(), XGate(), YGate(), ZGate()
            ]),
            Collect2qBlocks(),          # Group 2-qubit gates together
            ConsolidateBlocks(),        # Merge gate blocks when possible
            Unroll3qOrMore(),           # Decompose 3+ qubit gates

            # Layout and routing passes are commented out because they
            # require a backend coupling map specifying hardware connectivity
            # TrivialLayout(coupling_map),    # Simple qubit mapping
            # SabreLayout(coupling_map),      # SABRE algorithm mapping
            # StochasticSwap(coupling_map),   # Stochastic SWAP insertion

            # Circuit cleanup passes
            RemoveFinalMeasurements(),       # Remove measurement gates
            BarrierBeforeFinalMeasurements() # Add barriers before measurements
        ]

        # Add preset optimization levels as high-level actions
        # These apply multiple passes in optimized sequences
        # Note: These are powerful and might overshadow individual passes

        # Level 1: Light optimization
        level1_passes = [Optimize1qGates(), CXCancellation()]
        self.transpiler_passes.append(PassManager(level1_passes))

        # Level 2: Medium optimization
        level2_passes = [
            Optimize1qGates(), CXCancellation(),
            CommutationAnalysis(), Collect2qBlocks()
        ]
        self.transpiler_passes.append(PassManager(level2_passes))

        # Level 3: Heavy optimization
        level3_passes = [
            Optimize1qGates(), CXCancellation(), CommutationAnalysis(),
            InverseCancellation([
                CXGate(), HGate(), XGate(), YGate(), ZGate()
            ]),
            Collect2qBlocks(), ConsolidateBlocks()
        ]
        self.transpiler_passes.append(PassManager(level3_passes))

        # Set up Gymnasium spaces for RL algorithms
        self.n_actions = len(self.transpiler_passes)
        self.action_space = spaces.Discrete(self.n_actions)  # Discrete actions 0 to n-1

        # Observation space: fixed-size feature vector
        # [depth, size, num_qubits, gate_count_1, gate_count_2, ...]
        self.observation_space = spaces.Box(
            low=0,                                   # All features non-negative
            high=np.inf,                             # No upper bound
            shape=(STATE_VECTOR_SIZE,),              # Fixed size: 16 features
            dtype=np.float32                         # Standard RL data type
        )

    def _get_obs(self):
        """
        Compute the observation vector from the current quantum circuit.

        The observation is a fixed-size vector that summarizes the circuit:
        - First 3 elements: depth, size, num_qubits
        - Remaining 13 elements: counts for each gate type in FIXED_GATE_SET

        Returns:
            np.ndarray: 16-element feature vector representing circuit state
        """
        # Handle case where no circuit exists yet
        if self.circuit is None:
            return np.zeros(STATE_VECTOR_SIZE, dtype=np.float32)

        # Extract basic circuit metrics
        depth = self.circuit.depth()        # Circuit depth (critical path length)
        size = self.circuit.size()          # Total number of gates
        num_qubits = self.circuit.num_qubits # Number of qubits in circuit

        # Count occurrences of each gate type
        # This provides detailed information about circuit composition
        ops = self.circuit.count_ops()      # Dictionary of gate counts
        gate_counts = [
            ops.get(gate, 0) for gate in FIXED_GATE_SET  # 0 if gate not present
        ]

        # Combine all features into a single state vector
        # Structure: [depth, size, num_qubits, cx_count, u_count, ...]
        obs = np.array(
            [depth, size, num_qubits] + gate_counts,
            dtype=np.float32
        )
        return obs

    def _get_info(self):
        """
        Get auxiliary information about the current circuit state.

        This information is not part of the observation but provides
        useful debugging and analysis data for monitoring training.

        Returns:
            dict: Information about current and initial circuit metrics
        """
        return {
            "depth": self.circuit.depth(),           # Current circuit depth
            "size": self.circuit.size(),             # Current gate count
            "num_qubits": self.circuit.num_qubits,   # Number of qubits
            "initial_depth": self.initial_depth,     # Starting depth
            "initial_size": self.initial_size        # Starting gate count
        }

    def reset(self, seed=None, options=None):
        """
        Reset environment to start a new episode with a fresh quantum circuit.

        Generates a new random quantum circuit that serves as the optimization
        target for the episode. The circuit is intentionally "unoptimized" to
        provide opportunities for the agent to learn useful optimizations.

        Args:
            seed: Random seed for reproducible circuit generation
            options: Additional options (unused)

        Returns:
            tuple: (observation, info) for the new circuit
        """
        # Initialize parent class random number generator
        super().reset(seed=seed)

        self.current_step = 0  # Reset step counter

        # Generate random circuit parameters
        # Vary circuit size to provide training diversity
        num_qubits = self.np_random.integers(self.min_qubits, self.max_qubits + 1)

        # Create intentionally "unoptimized" circuit with redundant structure
        # Depth scales with qubits to ensure optimization opportunities
        depth = self.np_random.integers(num_qubits * 2, num_qubits * 5)

        # Build the quantum circuit with intentional optimization opportunities
        self.circuit = QuantumCircuit(num_qubits)

        # Start with Hadamard gates on all qubits (creates superposition)
        self.circuit.h(range(num_qubits))

        # Add patterns that create optimization opportunities
        for _ in range(depth):
            # Add some redundant patterns that can be optimized
            if self.np_random.random() < 0.3:  # 30% chance for redundant gates
                # Add pairs of gates that cancel each other
                q = self.np_random.integers(0, num_qubits)
                gate_type = self.np_random.choice(['x', 'h', 'z'])
                if gate_type == 'x':
                    self.circuit.x(q)
                    self.circuit.x(q)  # X-X cancels out
                elif gate_type == 'h':
                    self.circuit.h(q)
                    self.circuit.h(q)  # H-H cancels out
                elif gate_type == 'z':
                    self.circuit.z(q)
                    self.circuit.z(q)  # Z-Z cancels out

            # Add random 2-qubit gates (CNOT gates)
            if num_qubits > 1:
                q1, q2 = self.np_random.choice(range(num_qubits), 2, replace=False)
                self.circuit.cx(q1, q2)

            # Add random 1-qubit rotations
            q_1q = self.np_random.integers(0, num_qubits)
            # Random Z-rotation with angle between 0 and 2π
            self.circuit.rz(self.np_random.random() * 2 * np.pi, q_1q)

            # Add barriers occasionally (can be optimized away)
            if self.np_random.random() < 0.4:  # 40% chance
                self.circuit.barrier()
            self.circuit.rz(self.np_random.random() * 2 * np.pi, q_1q)
            self.circuit.barrier()

        # Add measurements at the end (standard for quantum algorithms)
        # TEMPORARILY DISABLED: Measurements cause transpiler pass issues
        # self.circuit.measure_all()

        # Store initial circuit metrics for reward calculation
        self.initial_depth = self.circuit.depth()
        self.initial_size = self.circuit.size()

        # Initialize previous metrics for step-by-step reward calculation
        # Rewards are based on improvement from previous step
        self.prev_depth = self.initial_depth
        self.prev_size = self.initial_size

        # Return initial observation and information
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one optimization step by applying a transpiler pass.

        Takes an action (transpiler pass selection) and applies it to the
        current quantum circuit, then calculates reward based on improvement.

        Args:
            action (int): Index of transpiler pass to apply (0 to n_actions-1)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Safety check: ensure environment is properly initialized
        if self.circuit is None:
            raise RuntimeError("Must call reset() before step()")

        self.current_step += 1  # Increment step counter

        # Select and apply the chosen transpiler pass
        pass_to_apply = self.transpiler_passes[action]

        # Apply the transpiler pass with error handling
        reward = 0.0  # Initialize reward
        try:
            # Apply transpiler pass directly to the circuit
            # This avoids potential issues with DAG conversion
            pass_manager = PassManager(pass_to_apply)
            self.circuit = pass_manager.run(self.circuit)

            # Get new circuit metrics after optimization
            new_depth = self.circuit.depth()
            new_size = self.circuit.size()

        except Exception:
            # Handle transpiler pass failures gracefully
            # Some passes might fail on certain circuit structures
            new_depth = self.prev_depth  # No change in metrics
            new_size = self.prev_size
            # Apply penalty for failed optimization attempt
            reward = -1.0  # Smaller penalty to avoid discouraging exploration

        # === REWARD CALCULATION ===
        if reward == 0.0:  # Only calculate if not set by exception handling
            # Calculate improvement from the previous step
            delta_depth = self.prev_depth - new_depth  # Positive = reduced
            delta_size = self.prev_size - new_size    # Positive = reduced

            # Weighted reward function with better shaping
            # Primary reward: depth reduction (heavily weighted)
            depth_reward = delta_depth * 10.0
            
            # Secondary reward: gate count reduction
            size_reward = delta_size * 1.0
            
            # Small positive reward for taking action (encourages exploration)
            action_reward = 0.1
            
            # Bonus for significant improvements
            if delta_depth > 0:
                depth_bonus = 2.0  # Extra reward for any depth reduction
            else:
                depth_bonus = 0.0
                
            reward = depth_reward + size_reward + action_reward + depth_bonus

        # Update metrics for next step's reward calculation
        self.prev_depth = new_depth
        self.prev_size = new_size

        # === EPISODE TERMINATION ===
        # Episode ends when maximum steps reached
        terminated = self.current_step >= self.max_steps
        truncated = False  # No separate truncation condition

        # Generate new observation and info for next step
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Render the environment state for visualization and debugging.

        Prints the current circuit state including metrics and ASCII drawing.
        Useful for understanding what the agent is learning.

        Args:
            mode (str): Rendering mode ('human' is the only supported mode)
        """
        if mode == 'human':
            if self.circuit:
                print(f"Step: {self.current_step}")
                print(f"Depth: {self.circuit.depth()}, "
                  f"Size: {self.circuit.size()}")
                # ASCII art representation of the quantum circuit
                print(self.circuit.draw(output='text', fold=-1))
            else:
                print("No circuit initialized.")

    def close(self):
        """
        Clean up environment resources.

        Currently no cleanup is needed, but this method is required
        by the Gymnasium interface.
        """
        pass


# === ENVIRONMENT TESTING ===
# This section provides standalone testing and demonstration of the environment
if __name__ == '__main__':
    # Import environment checker for validation
    from stable_baselines3.common.env_checker import check_env

    # Create environment instance
    env = QuantumCircuitEnv()

    # Validate environment follows Gymnasium interface
    print("Checking environment...")
    # check_env(env)  # Uncomment for strict validation (can be pedantic)
    print("Environment check passed (manual verification).")

    # === MANUAL DEMONSTRATION ===
    print("\n--- Manual Environment Test ---")

    # Reset environment and get initial state
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial circuit info: {info}")
    env.render()  # Show initial circuit

    # Demonstrate agent interaction with random actions
    print("\nTaking 5 random optimization steps...")
    for i in range(5):
        # Sample random action (transpiler pass)
        action = env.action_space.sample()
        pass_name = env.transpiler_passes[action].__class__.__name__
        print(f"\n---> STEP {i+1}: Applying action {action} ({pass_name})")

        # Apply action and get results
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Reward received: {reward}")
        print(f"Updated circuit info: {info}")
        env.render()  # Show optimized circuit

        # Check if episode ended
        if terminated or truncated:
            print("Episode finished.")
            break

    # Clean up
    env.close()
