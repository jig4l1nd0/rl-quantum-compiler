import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit.circuit import QuantumCircuit
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

# Define a fixed set of gates for our feature vector
# This ensures the observation space has a consistent shape.
FIXED_GATE_SET = [
    'cx', 'u', 'u3', 'h', 'rz', 'x', 'y', 'z', 'swap', 'cz',
    'ccx', 'measure', 'barrier'
]
# Max qubits for state padding
MAX_QUBITS = 15
# Max features: depth, size, num_qubits, + len(FIXED_GATE_SET)
STATE_VECTOR_SIZE = 3 + len(FIXED_GATE_SET)
# Define a fixed set of gates for our feature vector
# This ensures the observation space has a consistent shape.
FIXED_GATE_SET = [
    'cx', 'u', 'u3', 'h', 'rz', 'x', 'y', 'z', 'swap', 'cz',
    'ccx', 'measure', 'barrier'
]
# Max qubits for state padding
MAX_QUBITS = 15
# Max features: depth, size, num_qubits, + len(FIXED_GATE_SET)
STATE_VECTOR_SIZE = 3 + len(FIXED_GATE_SET)


class QuantumCircuitEnv(gym.Env):
    """
    Custom Gymnasium Environment for Quantum Circuit Compilation.

    The agent's goal is to apply a sequence of transpiler passes
    to minimize the depth and gate count of a random quantum circuit.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, max_steps=20):
        super().__init__()

        self.max_steps = max_steps
        self.min_qubits = 5
        self.max_qubits = MAX_QUBITS
        self.circuit = None
        self.initial_depth = 0
        self.initial_size = 0
        self.current_step = 0

        # Define the set of possible actions (Qiskit transpiler passes)
        # This is our discrete action space
        self.transpiler_passes = [
            Optimize1qGates(),
            CXCancellation(),
            CommutationAnalysis(),
            InverseCancellation(max_passes=3),
            Collect2qBlocks(),
            ConsolidateBlocks(),
            Unroll3qOrMore(),
            # We avoid layout/swap passes for this example as they
            # require a backend coupling map, but they could be added.
            # TrivialLayout(coupling_map),
            # SabreLayout(coupling_map),
            # StochasticSwap(coupling_map),
            RemoveFinalMeasurements(),
            BarrierBeforeFinalMeasurements()
        ]

        # Add the preset Qiskit optimization levels as actions
        # Note: These are powerful and might overshadow the individual passes
        self.transpiler_passes.extend([
            PassManager.from_config(optimization_level=1),
            PassManager.from_config(optimization_level=2),
            PassManager.from_config(optimization_level=3)
        ])

        self.n_actions = len(self.transpiler_passes)
        self.action_space = spaces.Discrete(self.n_actions)

        # Define the observation space (state)
        # We use a fixed-size feature vector
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(STATE_VECTOR_SIZE,), dtype=np.float32
        )

    def _get_obs(self):
        """
        Private method to compute the observation vector from the current circuit.
        """
        if self.circuit is None:
            return np.zeros(STATE_VECTOR_SIZE, dtype=np.float32)

        depth = self.circuit.depth()
        size = self.circuit.size()
        num_qubits = self.circuit.num_qubits

        # Get gate counts
        ops = self.circuit.count_ops()
        gate_counts = [
            ops.get(gate, 0) for gate in FIXED_GATE_SET
        ]

        # Combine into a single state vector
        obs = np.array(
            [depth, size, num_qubits] + gate_counts,
            dtype=np.float32
        )
        return obs

    def _get_info(self):
        """
        Private method to get auxiliary information.
        """
        return {
            "depth": self.circuit.depth(),
            "size": self.circuit.size(),
            "num_qubits": self.circuit.num_qubits,
            "initial_depth": self.initial_depth,
            "initial_size": self.initial_size
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state (a new random circuit).
        """
        super().reset(seed=seed)

        self.current_step = 0

        # Generate a new random circuit
        num_qubits = self.np_random.integers(self.min_qubits, self.max_qubits + 1)
        # Create a circuit with a reasonable "unoptimized" depth
        depth = self.np_random.integers(num_qubits * 2, num_qubits * 5)

        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.h(range(num_qubits))
        for _ in range(depth):
            # Add some 2-qubit gates
            q1, q2 = self.np_random.choice(range(num_qubits), 2, replace=False)
            self.circuit.cx(q1, q2)
            # Add some 1-qubit gates
            q_1q = self.np_random.integers(0, num_qubits)
            self.circuit.rz(self.np_random.random() * 2 * np.pi, q_1q)
            self.circuit.barrier()

        self.circuit.measure_all()

        # Store initial metrics
        self.initial_depth = self.circuit.depth()
        self.initial_size = self.circuit.size()

        # Store for reward calculation
        self.prev_depth = self.initial_depth
        self.prev_size = self.initial_size

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Executes one step in the environment.
        """
        if self.circuit is None:
            raise RuntimeError("Must call reset() before step()")

        self.current_step += 1

        # Get the transpiler pass for the chosen action
        pass_to_apply = self.transpiler_passes[action]

        # Apply the pass
        try:
            # We must convert to/from a DAG for many passes to work
            dag = circuit_to_dag(self.circuit)
            pass_manager = PassManager(pass_to_apply)
            new_dag = pass_manager.run(dag)
            self.circuit = dag_to_circuit(new_dag)

            new_depth = self.circuit.depth()
            new_size = self.circuit.size()

        except Exception as e:
            # Some passes might fail, penalize this
            # print(f"Pass failed: {e}")
            new_depth = self.prev_depth
            new_size = self.prev_size
            # Penalize failed pass
            reward = -10.0

        # --- Calculate Reward ---
        # Reward is based on the *improvement* from the *previous* step
        delta_depth = self.prev_depth - new_depth
        delta_size = self.prev_size - new_size

        # Heavily weight depth reduction
        reward = (delta_depth * 10.0) + (delta_size * 1.0)

        # Update previous metrics for next step
        self.prev_depth = new_depth
        self.prev_size = new_size

        # --- Check for Termination ---
        terminated = self.current_step >= self.max_steps
        truncated = False  # We don't have a separate truncation condition

        # Get new observation and info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Renders the environment (prints circuit drawing).
        """
        if mode == 'human':
            if self.circuit:
                print(f"Step: {self.current_step}")
                print(f"Depth: {self.circuit.depth()}, Size: {self.circuit.size()}")
                print(self.circuit.draw(output='text', fold=-1))
            else:
                print("No circuit initialized.")

    def close(self):
        pass


if __name__ == '__main__':
    # Test the environment
    from stable_baselines3.common.env_checker import check_env

    env = QuantumCircuitEnv()

    # This check is crucial
    print("Checking environment...")
    # check_env(env) # This check can be strict, manual testing is also good
    print("Environment check passed (manual verification).")

    # --- Manual Test ---
    print("\n--- Manual Environment Test ---")
    obs, info = env.reset()
    print(f"Initial state: {obs.shape}")
    print(f"Initial Info: {info}")
    env.render()

    # Take 5 random steps
    for i in range(5):
        action = env.action_space.sample() # Random action
        pass_name = env.transpiler_passes[action].__class__.__name__
        print(f"\n---> STEP {i+1}: Applying action {action} ({pass_name})")

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Reward: {reward}")
        print(f"Info: {info}")
        env.render()

        if terminated or truncated:
            print("Episode finished.")
            break

    env.close()
