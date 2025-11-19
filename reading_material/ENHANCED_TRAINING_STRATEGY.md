# Enhanced RL Training Strategy for Action Diversity

## Problem Analysis
The current PPO agent converges to Action 3 (CommutationAnalysis) for all scenarios because:
1. **Training data bias**: Random circuits likely favored commutation-based optimizations
2. **Reward function**: Action 3 provided consistent positive rewards across diverse circuits  
3. **Exploration decay**: PPO exploration decreased over training, converging to safe action
4. **Action imbalance**: Some actions (barriers, resets) are rare in random circuits

## Solution 1: Curriculum Learning with Targeted Circuits

### Generate Action-Specific Training Sets
```bash
python training_circuits_generator.py
```

This creates 50+ circuits for each action type:
- **CXCancellation**: CNOT pairs that should cancel
- **ConsolidateBlocks**: Multiple single-qubit rotations 
- **InverseCancellation**: H-H, X-X, S-Sdg pairs
- **Optimize1qGates**: Inefficient single-qubit sequences
- **RemoveBarriers**: Circuits with unnecessary barriers
- **RemoveResetInZeroState**: Redundant reset operations

### Balanced Training Protocol
```python
# Enhanced training with action-specific episodes
total_episodes = 500_000  # Increased from original
action_episodes = total_episodes // 14  # ~35,714 per action

for action_type in action_types:
    for episode in range(action_episodes):
        # Use circuits designed for this specific action
        circuit = sample_circuit_for_action(action_type)
        train_episode(circuit)
```

## Solution 2: Modified Reward Function

### Current Reward Issues
- Heavily favors depth reduction (good)
- Doesn't encourage action diversity
- No penalty for repeated ineffective actions

### Enhanced Reward Function
```python
def calculate_enhanced_reward(prev_depth, new_depth, prev_size, new_size, action_history):
    # Base reward (existing logic)
    base_reward = calculate_base_reward(prev_depth, new_depth, prev_size, new_size)
    
    # Action diversity bonus
    recent_actions = action_history[-5:]  # Last 5 actions
    if len(set(recent_actions)) > 1:
        diversity_bonus = 0.5  # Small bonus for action variety
    else:
        diversity_bonus = 0
    
    # Repeated action penalty
    if len(recent_actions) >= 3 and len(set(recent_actions)) == 1:
        repetition_penalty = -0.2  # Discourage action repetition
    else:
        repetition_penalty = 0
    
    return base_reward + diversity_bonus + repetition_penalty
```

## Solution 3: Multi-Action Training Episodes

### Force Action Exploration
```python
class ActionBalancedEnv(QuantumCircuitEnv):
    def __init__(self):
        super().__init__()
        self.action_counts = [0] * 14  # Track action usage
        
    def step(self, action):
        self.action_counts[action] += 1
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Encourage underused actions
        min_usage = min(self.action_counts)
        if self.action_counts[action] <= min_usage + 5:
            reward += 0.3  # Bonus for balanced exploration
            
        return obs, reward, terminated, truncated, info
```

## Solution 4: Experience Replay with Action Balancing

### Prioritized Experience Replay
```python
# Store successful episodes for each action
action_memory = {action_id: [] for action_id in range(14)}

def store_experience(episode_data, successful_actions):
    for action in successful_actions:
        action_memory[action].append(episode_data)

def balanced_replay_sampling():
    # Sample equally from each action's memory
    experiences = []
    for action_memory_list in action_memory.values():
        if action_memory_list:
            experiences.extend(random.sample(action_memory_list, min(10, len(action_memory_list))))
    return experiences
```

## Solution 5: Ensemble of Specialized Agents

### Multiple Expert Policies
Instead of one general agent, train specialists:
```python
specialized_agents = {
    'cancellation_expert': train_agent(cx_cancellation_circuits),
    'consolidation_expert': train_agent(consolidation_circuits), 
    'cleanup_expert': train_agent(barrier_reset_circuits),
    'general_expert': train_agent(mixed_circuits)
}

def predict_with_ensemble(circuit):
    # Use circuit features to select best expert
    if has_cancelling_gates(circuit):
        return cancellation_expert.predict(circuit)
    elif has_many_single_qubit_gates(circuit):
        return consolidation_expert.predict(circuit)
    # etc.
```

## Implementation Priority

### Phase 1: Quick Fix (1-2 hours)
1. **Generate action-specific circuits** using the generator script
2. **Retrain with balanced dataset** (50k episodes with equal action representation)
3. **Test action diversity** with the enhanced circuits

### Phase 2: Advanced Enhancement (1 day)
1. **Implement enhanced reward function** with action diversity incentives  
2. **Add action history tracking** to the environment
3. **Retrain with new reward structure**

### Phase 3: Production Enhancement (2-3 days)
1. **Implement ensemble approach** with specialized agents
2. **Create circuit classifier** to route to appropriate expert
3. **Deploy multi-agent system**

## Expected Outcomes

### After Phase 1:
- **Action usage should spread** across multiple actions (not just Action 3)
- **Specialized optimizations** for barriers, resets, cancellations
- **Maintained performance** on existing Action 3 scenarios

### After Phase 2:
- **Dynamic action selection** based on circuit characteristics
- **Reduced action repetition** when ineffective
- **Improved overall optimization** across diverse circuit types

### After Phase 3:
- **Expert-level optimization** for each transpiler pass type
- **Intelligent routing** to most appropriate optimization strategy
- **Production-ready multi-agent system**

Would you like me to implement any of these solutions? I recommend starting with **Phase 1** using the circuit generator to create a more balanced training dataset.