# Theoretical Background: Quantum Computing & Reinforcement Learning

## Overview
This document provides the theoretical foundation necessary to understand the RL Quantum Circuit Optimizer project, covering both quantum computing and reinforcement learning concepts.

---

## 1. Quantum Computing Fundamentals

### 1.1 Quantum Bits (Qubits)
- **Classical Bit**: Can be in state 0 or 1
- **Qubit**: Can be in superposition of both states: 
  $$|ψ⟩ = α|0⟩ + β|1⟩$$
  where $|α|^2 + |β|^2 = 1$ (normalization condition)
- **Bloch Sphere**: Geometric representation where any qubit state is:
  $$|ψ⟩ = \cos(\frac{θ}{2})|0⟩ + e^{iφ}\sin(\frac{θ}{2})|1⟩$$
- **Measurement**: Collapses superposition to classical state with probabilities $P(0) = |α|^2$, $P(1) = |β|^2$

**Example**: Equal superposition state $|+⟩ = \frac{1}{\sqrt{2}}(|0⟩ + |1⟩)$ has 50% chance of measuring 0 or 1.

### 1.2 Quantum Gates
**Single-Qubit Gates:**
- **Pauli Gates**: 
  - X (bit-flip): $X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $X|0⟩ = |1⟩$, $X|1⟩ = |0⟩$
  - Y (bit and phase flip): $Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$
  - Z (phase-flip): $Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$, $Z|0⟩ = |0⟩$, $Z|1⟩ = -|1⟩$

- **Hadamard (H)**: Creates superposition
  $$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$
  $$H|0⟩ = \frac{1}{\sqrt{2}}(|0⟩ + |1⟩) = |+⟩$$
  $$H|1⟩ = \frac{1}{\sqrt{2}}(|0⟩ - |1⟩) = |-⟩$$

- **Phase Gates**: 
  - S gate: $S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$
  - T gate: $T = \begin{pmatrix} 1 & 0 \\ 0 & e^{iπ/4} \end{pmatrix}$
  - Rotation: $R_z(θ) = \begin{pmatrix} e^{-iθ/2} & 0 \\ 0 & e^{iθ/2} \end{pmatrix}$

- **Universal Gate**: U3(θ,φ,λ) - any single-qubit rotation
  $$U_3(θ,φ,λ) = \begin{pmatrix} \cos(θ/2) & -e^{iλ}\sin(θ/2) \\ e^{iφ}\sin(θ/2) & e^{i(φ+λ)}\cos(θ/2) \end{pmatrix}$$

**Two-Qubit Gates:**
- **CNOT (CX)**: Controlled-X, creates entanglement
  $$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$
  Example: $\text{CNOT}|00⟩ = |00⟩$, $\text{CNOT}|10⟩ = |11⟩$

- **CZ**: Controlled-Z gate: $\text{CZ}|11⟩ = -|11⟩$, others unchanged
- **SWAP**: Exchanges qubit states: $\text{SWAP}|01⟩ = |10⟩$

**Multi-Qubit Gates:**
- **Toffoli (CCX)**: Controlled-controlled-X (3-qubit gate)
- **Fredkin**: Controlled-SWAP

**Example Circuit:**
```
     ┌───┐     ┌─┐
q_0: ┤ H ├──■──┤M├
     └───┘┌─┴─┐└╥┘
q_1: ─────┤ X ├─╫─
          └───┘ ║ 
c: 1/═══════════╩═
                0 
```
Creates Bell state: $\frac{1}{\sqrt{2}}(|00⟩ + |11⟩)$

### 1.3 Quantum Circuits
- **Circuit Depth**: Maximum number of gate layers (parallel gates count as one layer)
- **Circuit Size**: Total number of gates in the circuit
- **QASM**: Quantum Assembly Language for circuit representation
- **Transpilation**: Converting logical circuits to physical hardware constraints

### 1.4 Quantum Circuit Optimization
**Optimization Goals:**
- Reduce circuit depth (critical for NISQ devices due to decoherence)
- Minimize gate count (reduce noise accumulation: $p_{error} ≈ n_{gates} \times p_{gate}$)
- Hardware compatibility (native gate sets)

**Common Optimization Techniques:**

**1. Gate Cancellation**: Adjacent inverse gates cancel out
```
Example: X-X = I (identity)
     ┌───┐┌───┐       
q_0: ┤ X ├┤ X ├  →  q_0: ──────
     └───┘└───┘           
```

**2. Commutation Analysis**: Reorder commuting gates for better structure
```
Gates commute if [A,B] = AB - BA = 0
Example: Z gates on different qubits commute
     ┌───┐ ┌───┐     ┌───┐ ┌───┐
q_0: ┤ Z ├─┤ H ├  =  ┤ H ├─┤ Z ├  (if operations are independent)
     └───┘ └───┘     └───┘ └───┘
```

**3. Template Matching**: Replace gate sequences with equivalent shorter ones
```
Example: H-X-H = Z (basis rotation identity)
     ┌───┐┌───┐┌───┐     ┌───┐
q_0: ┤ H ├┤ X ├┤ H ├  →  ┤ Z ├
     └───┘└───┘└───┘     └───┘
```

**4. Heuristic Synthesis**: Use optimization heuristics for gate reduction

**Circuit Quality Metrics:**
- **Depth**: $d = \max_i(\text{layer}(g_i))$ where $g_i$ are gates
- **Size**: $s = \sum_i 1$ (total gate count)
- **Fidelity**: $F = |\langle ψ_{ideal}|ψ_{optimized}⟩|^2$ (should be ≈ 1)

---

## 2. Reinforcement Learning Fundamentals

### 2.1 Basic RL Framework
**Agent-Environment Interaction:**
```
Agent → Action (a_t) → Environment
  ↑                        ↓
Reward (r_t) ← State (s_t+1)
```

**Mathematical Formulation:**
At each time step $t$:
- Agent observes state $s_t \in S$
- Selects action $a_t \sim π(a|s_t)$ according to policy $π$
- Environment transitions to $s_{t+1} \sim P(s'|s_t, a_t)$
- Agent receives reward $r_t = R(s_t, a_t, s_{t+1})$

**Key Components:**
- **State (s)**: Environment configuration, $s_t \in S$
- **Action (a)**: Agent's decision from action space $A$, $a_t \in A$
- **Reward (r)**: Feedback signal $r_t \in \mathbb{R}$
- **Policy (π)**: Strategy $π: S \times A → [0,1]$ where $π(a|s) = P(A_t=a|S_t=s)$
- **Value Function (V)**: Expected cumulative reward
  $$V^π(s) = \mathbb{E}_π\left[\sum_{t=0}^∞ γ^t R_{t+1} \Big| S_0 = s\right]$$

**Example**: In our quantum optimizer:
- $s_t$: Circuit representation [depth, size, gate_counts...]
- $a_t$: Transpiler pass selection (0-13)
- $r_t$: Circuit improvement measure
- $π(a|s)$: Learned policy network output

### 2.2 Markov Decision Process (MDP)
**Mathematical Framework:**
MDP is defined by tuple $(S, A, P, R, γ)$ where:
- **States**: $S$ (all possible environment states)
- **Actions**: $A$ (all possible agent actions)
- **Transition Probabilities**: $P(s'|s,a) = \Pr(S_{t+1}=s'|S_t=s, A_t=a)$
- **Reward Function**: $R(s,a,s') = \mathbb{E}[R_{t+1}|S_t=s, A_t=a, S_{t+1}=s']$
- **Discount Factor**: $γ \in [0,1]$ (future reward importance)

**Markov Property**: 
$$P(S_{t+1}=s'|S_t=s_t, A_t=a_t, S_{t-1}=s_{t-1}, ...) = P(S_{t+1}=s'|S_t=s_t, A_t=a_t)$$

**Bellman Equations:**
- **State Value**: $V^π(s) = \sum_a π(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]$
- **Action Value**: $Q^π(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]$
- **Optimal Value**: $V^*(s) = \max_a Q^*(s,a)$

**Example**: Quantum Circuit Optimization MDP
- $S$: All possible quantum circuit configurations
- $A$: {Unroller, CXCancellation, CommutationAnalysis, ...} (14 transpiler passes)
- $P(s'|s,a)$: Deterministic (applying transpiler pass to circuit)
- $R(s,a,s')$: Circuit improvement metric
- $γ = 0.99$: High discount (long-term optimization)

### 2.3 Policy-Based Methods
**Policy Gradient Methods:**
- **Objective**: Maximize expected cumulative reward 
  $$J(π_θ) = \mathbb{E}_{τ \sim π_θ}\left[\sum_{t=0}^T γ^t r_t\right] = \mathbb{E}_{s \sim d^π}[V^π(s)]$$
  where $τ = (s_0, a_0, r_0, s_1, a_1, r_1, ...)$ is a trajectory

- **Policy Gradient Theorem**: 
  $$\nabla_θ J(π_θ) = \mathbb{E}_{τ \sim π_θ}\left[\sum_{t=0}^T \nabla_θ \log π_θ(a_t|s_t) \cdot A^π(s_t,a_t)\right]$$

- **Advantage Function**: $A^π(s,a) = Q^π(s,a) - V^π(s)$
  - Measures how much better action $a$ is compared to average in state $s$
  - Reduces variance in policy gradient estimates

**REINFORCE Algorithm:**
```python
# Simplified REINFORCE pseudocode
for episode in range(num_episodes):
    trajectory = collect_trajectory(policy)
    for t in range(len(trajectory)):
        G_t = sum(gamma^k * rewards[t+k] for k in range(len(rewards)-t))
        gradient += log_prob[t] * G_t
    policy.update(gradient)
```

**Example**: In quantum optimization
- $θ$: Neural network parameters
- $π_θ(a|s)$: Softmax output over 14 transpiler actions
- $A(s,a)$: How much better this transpiler pass is than average
- High variance requires baseline subtraction: $A(s,a) = Q(s,a) - V(s)$

### 2.4 Actor-Critic Methods
**Architecture:**
- **Actor**: Policy network π_θ(a|s) (action selection)
- **Critic**: Value network V_φ(s) (state evaluation)
- **Advantage**: A(s,a) = r + γV(s') - V(s)

**Benefits:**
- Lower variance than pure policy gradient
- More stable learning
- Better sample efficiency

---

## 3. Proximal Policy Optimization (PPO)

### 3.1 PPO Algorithm
**Problem with Basic Policy Gradient:**
Large policy updates can be destructive. If $π_{new}(a|s) >> π_{old}(a|s)$ or $π_{new}(a|s) << π_{old}(a|s)$, training becomes unstable.

**PPO Clipped Objective:**
$$L^{CLIP}(θ) = \mathbb{E}_t\left[\min\left(r_t(θ)A_t, \text{clip}(r_t(θ), 1-ε, 1+ε)A_t\right)\right]$$

Where:
- **Probability Ratio**: $r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{θ_{old}}(a_t|s_t)}$
- **Clipping Range**: $ε ≈ 0.2$ (typical value)
- **Advantage**: $A_t = \sum_{l=0}^∞ (γλ)^l δ_{t+l}$ (GAE)
- **TD Error**: $δ_t = r_t + γV(s_{t+1}) - V(s_t)$

**Clipping Mechanism:**
- If $A_t > 0$ (good action): clip $r_t(θ)$ to $(1, 1+ε]$
- If $A_t < 0$ (bad action): clip $r_t(θ)$ to $[1-ε, 1)$
- Prevents too large policy updates

**Complete PPO Loss:**
$$L(θ) = \mathbb{E}_t\left[L^{CLIP}(θ) - c_1 L^{VF}(θ) + c_2 S[π_θ](s_t)\right]$$

Where:
- $L^{VF}(θ) = (V_θ(s_t) - V_t^{target})^2$ (value function loss)
- $S[π_θ](s_t) = -\sum_a π_θ(a|s_t) \log π_θ(a|s_t)$ (entropy bonus)
- $c_1 = 0.5$, $c_2 = 0.01$ (typical coefficients)

**Example Calculation:**
```python
# If old_prob = 0.1, new_prob = 0.3, advantage = 2.0, epsilon = 0.2
ratio = 0.3 / 0.1 = 3.0
clipped_ratio = clip(3.0, 0.8, 1.2) = 1.2
loss = min(3.0 * 2.0, 1.2 * 2.0) = min(6.0, 2.4) = 2.4
# Without clipping, loss would be 6.0 (too aggressive)
```

### 3.2 PPO Implementation Details
**Network Architecture:**
- Shared layers for feature extraction
- Separate heads for policy and value
- Typical: Fully connected layers with ReLU activation

**Training Process:**
1. Collect trajectories with current policy
2. Compute advantages using GAE (Generalized Advantage Estimation)
3. Update policy with clipped objective
4. Update value function with MSE loss
5. Repeat until convergence

---

## 4. Application to Quantum Circuit Optimization

### 4.1 RL Environment Design
**State Representation:**
Our environment encodes quantum circuits as fixed-size observation vectors:

$$\mathbf{s} = [d, g, n, c_{cx}, c_u, c_{u3}, c_h, c_{rz}, c_x, c_y, c_z, c_{swap}, c_{cz}, c_{ccx}, c_{measure}, c_{barrier}]$$

Where:
- $d$ = circuit depth
- $g$ = total gate count  
- $n$ = number of qubits
- $c_i$ = count of gate type $i$

**Example State Vector:**
```python
# For circuit: H-CX-RZ-MEASURE on 2 qubits
state = [2,    # depth
         4,    # total gates (H + CX + RZ + MEASURE)
         2,    # qubits
         1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#        cx u  u3 h  rz x  y  z  sw cz ccx meas bar
```

**Action Space:**
Discrete action space $\mathcal{A} = \{0, 1, 2, ..., 13\}$ where each action applies a specific transpiler pass:

$$a_t \in \mathcal{A} \rightarrow \text{Transpiler Pass Application}$$

**Reward Function:**
Multi-objective reward combining depth and gate reduction:

$$R(s,a,s') = w_d \cdot \frac{d - d'}{d} + w_g \cdot \frac{g - g'}{g} + R_{diversity} + R_{penalty}$$

Where:
- $w_d = 2.0$ (depth weight, critical for NISQ)
- $w_g = 1.0$ (gate weight)  
- $R_{diversity} = +0.2$ (if new action used)
- $R_{penalty} = -0.3$ (if repeating recent action)

**Example Reward Calculation:**
```python
# Circuit: depth 10→8, gates 20→15
depth_improvement = (10 - 8) / 10 = 0.2
gate_improvement = (20 - 15) / 20 = 0.25
base_reward = 2.0 * 0.2 + 1.0 * 0.25 = 0.65

# If using diverse action: reward = 0.65 + 0.2 = 0.85
# If repeating action: reward = 0.65 - 0.3 = 0.35
```

### 4.2 Action Mapping
**Transpiler Passes as Actions:**
```python
ACTIONS = {
    0: Unroller(['cx', 'u3']),           # Basic gate decomposition
    1: CXCancellation(),                 # Cancel adjacent CX gates
    2: CommutationAnalysis(),            # Analyze gate commutation
    3: CommutativeCancellation(),        # Cancel commuting gates
    4: Optimize1qGates(),                # Single-qubit optimization
    5: RemoveBarriers(),                 # Remove barrier instructions
    6: InverseCancellation(),            # Cancel inverse operations
    7: HeuristicsSynthesis(),            # Heuristic optimization
    8: TemplateOptimization(),           # Template-based optimization
    # ... additional passes
}
```

### 4.3 Training Challenges & Solutions

**Challenge 1: Action Diversity**
- **Problem**: Model converges to single dominant action (Action 3 bias)
  $$P(a=3|s) \approx 1.0, \quad P(a \neq 3|s) \approx 0.0$$
- **Root Cause**: Action 3 (CommutationAnalysis) often provides safe positive rewards
- **Solution**: Enhanced reward with diversity incentives
  $$R_{enhanced}(s,a,s') = R_{base}(s,a,s') + R_{diversity}(a) + R_{penalty}(a, history)$$

**Challenge 2: Environment State Corruption**
- **Problem**: Custom circuit injection breaks internal environment state
- **Mathematical Issue**: $s_{observed} \neq s_{actual}$ after circuit injection
- **Solution**: Explicit state reconstruction after injection
  ```python
  # Proper state management
  env.reset()                    # Clean state
  env.circuit = custom_circuit   # Inject circuit
  env._update_internal_state()   # Reconstruct observation
  obs = env._get_obs()          # Correct observation
  ```

**Challenge 3: Reward Engineering**
- **Problem**: Sparse rewards lead to poor exploration
  $$R(s,a,s') = \begin{cases} 
  0 & \text{if no improvement} \\
  \text{large positive} & \text{if improvement}
  \end{cases}$$
- **Solution**: Dense rewards with intermediate feedback
  $$R(s,a,s') = \alpha \cdot \Delta d + \beta \cdot \Delta g + \gamma \cdot I_{valid} + \delta \cdot I_{diverse}$$

**Mathematical Formulation of Enhanced Training:**

**Diversity Tracking:**
$$H_t = [a_{t-k}, a_{t-k+1}, ..., a_{t-1}] \quad \text{(action history window)}$$

**Repetition Penalty:**
$$R_{penalty}(a_t, H_t) = \begin{cases}
-0.3 & \text{if } a_t \in H_t \\
0 & \text{otherwise}
\end{cases}$$

**Diversity Bonus:**
$$R_{diversity}(a_t, H_t) = \begin{cases}
+0.2 & \text{if } a_t \notin H_t \\
0 & \text{otherwise}
\end{cases}$$

**Action Balance Objective:**
$$\min_θ \sum_{i=0}^{13} \left(P_θ(a_i) - \frac{1}{14}\right)^2$$
Where $P_θ(a_i) = \mathbb{E}_{s \sim D}[π_θ(a_i|s)]$ over training distribution $D$.

### 4.4 Enhanced Training Strategy

**Curriculum Learning Mathematical Framework:**

**Training Set Composition:**
$$\mathcal{D}_{train} = \alpha \cdot \mathcal{D}_{specific} + (1-\alpha) \cdot \mathcal{D}_{random}$$

Where:
- $\mathcal{D}_{specific}$: Action-specific training circuits (30%, $\alpha = 0.3$)
- $\mathcal{D}_{random}$: Random quantum circuits (70%, $1-\alpha = 0.7$)

**Action-Specific Circuit Generation:**
For each action $a_i$, generate circuits $C_{a_i}$ where applying $a_i$ yields high reward:

$$C_{a_i} = \{c : \mathbb{E}[R(s_c, a_i, s'_c)] > \tau_{reward}\}$$

**Examples:**
1. **Action 6 (InverseCancellation)**: Circuits with inverse gate pairs
   ```
   Circuit: X-X-H  →  Expected: X cancellation  →  H
   Mathematical: XX = I, so circuit simplifies to H
   ```

2. **Action 7 (HeuristicsSynthesis)**: Complex multi-qubit circuits
   ```
   Circuit: Multi-CX patterns  →  Expected: Synthesis optimization
   Mathematical: Optimize CX count via synthesis algorithms
   ```

**Diversity Metrics During Training:**

**Action Distribution Entropy:**
$$H(π) = -\sum_{a=0}^{13} P(a) \log P(a)$$
Target: $H(π) \approx \log(14) = 2.64$ (uniform distribution)

**Gini Coefficient for Action Balance:**
$$G = \frac{1}{2n^2\bar{p}} \sum_{i=1}^n \sum_{j=1}^n |p_i - p_j|$$
Target: $G \approx 0$ (perfect equality)

**Training Objective with Balance:**
$$\mathcal{L}_{total} = \mathcal{L}_{PPO} + \lambda_{entropy} H(π) + \lambda_{balance} G$$

Where:
- $\lambda_{entropy} = 0.01$ (entropy regularization)
- $\lambda_{balance} = 0.1$ (action balance penalty)

**Curriculum Progression:**
$$P(\text{action-specific circuit}) = \min(0.3, 0.1 + 0.0001 \cdot \text{timestep})$$

Starting with 10% action-specific, gradually increasing to 30% maximum.

---

## 5. Performance Metrics & Evaluation

### 5.1 Circuit Quality Metrics
**Primary Metrics:**
- **Depth Reduction**: (initial_depth - final_depth) / initial_depth
- **Gate Reduction**: (initial_gates - final_gates) / initial_gates
- **Fidelity Preservation**: Circuit functional equivalence

**Secondary Metrics:**
- **Training Stability**: Reward convergence
- **Action Diversity**: Distribution across action space
- **Generalization**: Performance on unseen circuits

### 5.2 Evaluation Methodology
**Testing Protocol:**
1. Diverse circuit generation (multiple patterns)
2. Multiple optimization runs per circuit
3. Statistical analysis of results
4. Action usage pattern analysis
5. Comparison with baseline transpilers

**Success Criteria:**
- Consistent optimization across circuit types
- Action diversity > 2 primary actions
- Performance improvement over single-action baseline

---

## 6. Future Directions

### 6.1 Advanced RL Techniques
- **Multi-Agent RL**: Parallel optimization strategies
- **Hierarchical RL**: Multi-level optimization decisions
- **Meta-Learning**: Quick adaptation to new circuit types
- **Curriculum Learning**: Systematic difficulty progression

### 6.2 Quantum-Specific Enhancements
- **Hardware-Aware Training**: Device-specific constraints
- **Noise Modeling**: NISQ device error incorporation
- **Connectivity Constraints**: Physical qubit topology
- **Gate Fidelity Optimization**: Error-rate aware decisions

### 6.3 Scalability Considerations
- **Large Circuit Handling**: Efficient state representation
- **Distributed Training**: Multi-node RL training
- **Real-Time Optimization**: Low-latency deployment
- **Continuous Learning**: Online adaptation to new patterns

---

## Conclusion

The intersection of quantum computing and reinforcement learning creates a powerful framework for intelligent circuit optimization. This project demonstrates how RL agents can learn sophisticated optimization strategies that surpass traditional rule-based approaches, achieving genuine intelligence in quantum circuit manipulation.

**Key Insights:**
1. **State Design**: Effective quantum circuit representation in RL states
2. **Action Engineering**: Mapping optimization techniques to discrete actions
3. **Reward Shaping**: Balancing multiple optimization objectives
4. **Training Stability**: Ensuring diverse, robust policy learning

This theoretical foundation enables understanding of both the quantum optimization problem and the RL solution methodology, providing the context necessary to extend and improve the system further.