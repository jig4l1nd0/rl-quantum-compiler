"""
RL Quantum Circuit Compiler - FastAPI + Custom HTML Frontend

This module provides a web-based interface for optimizing quantum circuits using
a trained PPO reinforcement learning agent. The application serves a custom HTML
frontend with Tailwind CSS and provides RESTful API endpoints for circuit optimization.

Features:
- Custom HTML interface with Tailwind CSS styling
- RESTful API for quantum circuit optimization
- Support for QASM 2.0 input format  
- Real-time optimization metrics
- Production-ready deployment with health checks

API Endpoints:
- GET /: Serves the HTML frontend
- POST /api/optimize: Optimizes quantum circuits
- GET /health: Health check for deployment monitoring

Requirements:
- Trained PPO model file: 'ppo_quantum_compiler_enhanced.zip'
- FastAPI: Web framework and API server
- Stable Baselines3: RL model loading
- Qiskit: Quantum circuit manipulation
"""

import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from stable_baselines3 import PPO
from qiskit.circuit import QuantumCircuit
from qiskit.qasm2 import loads as qasm2_loads, dumps as qasm2_dumps

# Import the RL environment for model loading
from rl_compiler_env import QuantumCircuitEnv


# === API REQUEST/RESPONSE MODELS ===


class OptimizationRequest(BaseModel):
    """Request model for circuit optimization endpoint."""
    qasm_string: str


class OptimizationResponse(BaseModel):
    """Response model for circuit optimization results."""
    initial_qasm: str
    optimized_qasm: str
    initial_depth: int
    initial_size: int
    optimized_depth: int
    optimized_size: int
    depth_reduction_percent: float
    gate_reduction_percent: float
    total_steps: int


# === MODEL LOADING AND INITIALIZATION ===

# Configuration
MODEL_PATH = "ppo_quantum_compiler_enhanced.zip"
HTML_FILE = "index.html"

# Initialize global model variable
model = None
dummy_env = None


def load_model():
    """Load the trained PPO model with error handling."""
    global model, dummy_env

    try:
        if not os.path.exists(MODEL_PATH):
            print(f"WARNING: Model file {MODEL_PATH} not found.")
            print("Please run train_enhanced.py to generate the enhanced trained model.")
            return False

        print("Loading trained PPO model...")
        dummy_env = QuantumCircuitEnv()
        model = PPO.load(MODEL_PATH, env=dummy_env)
        print("_______Model loaded successfully.")
        return True

    except Exception as e:
        print(f"______Failed to load model: {e}")
        return False


# === FASTAPI APPLICATION SETUP ===


app = FastAPI(
    title="RL Quantum Circuit Compiler",
    description="Optimize quantum circuits using reinforcement learning",
    version="1.0.0"
)

# Load the model on startup
model_loaded = load_model()


# === API ENDPOINTS ===


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the custom HTML frontend interface.

    Returns:
        HTMLResponse: The main application interface
    """
    try:
        html_path = Path(HTML_FILE)
        if not html_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Frontend file {HTML_FILE} not found"
            )

        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        return HTMLResponse(content=html_content, status_code=200)

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error serving frontend: {str(e)}"
        )


@app.post("/api/optimize", response_model=OptimizationResponse)
async def optimize_circuit(request: OptimizationRequest) -> OptimizationResponse:
    """
    Optimize a quantum circuit using the trained RL agent.

    This endpoint takes a QASM 2.0 string, parses it into a quantum circuit,
    injects it into the RL environment, and runs the trained PPO agent to
    apply optimization passes.

    Args:
        request: OptimizationRequest containing the QASM string

    Returns:
        OptimizationResponse: Optimization results and metrics

    Raises:
        HTTPException: If model not loaded, invalid QASM, or optimization fails
    """
    # Check if model is available
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please ensure ppo_quantum_compiler_enhanced.zip exists and train_enhanced.py has been run."
        )

    # Validate input
    qasm_string = request.qasm_string.strip()
    if not qasm_string:
        raise HTTPException(
            status_code=400,
            detail="QASM input cannot be empty"
        )

    try:
        # Parse QASM string into quantum circuit
        circuit = qasm2_loads(qasm_string)

        # Extract initial circuit metrics
        initial_qasm = qasm2_dumps(circuit)
        initial_depth = circuit.depth()
        initial_size = circuit.size()

        if initial_depth == 0:
            raise HTTPException(
                status_code=400,
                detail="Circuit appears to be empty or invalid (depth = 0)"
            )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid QASM input: {str(e)}"
        )

    try:
        # Use environment directly instead of vectorized wrapper
        env_direct = QuantumCircuitEnv()
        
        # First reset to get clean environment state
        obs, info = env_direct.reset()
        
        # Then inject our custom circuit AFTER reset
        env_direct.circuit = circuit.copy()
        env_direct.initial_depth = initial_depth
        env_direct.initial_size = initial_size
        env_direct.prev_depth = initial_depth
        env_direct.prev_size = initial_size
        env_direct.current_step = 0

        # Manually create observation from the injected circuit
        # This replicates _get_obs logic but using public interfaces
        FIXED_GATE_SET = [
            'cx', 'u', 'u3', 'h', 'rz', 'x', 'y', 'z', 'swap', 'cz',
            'ccx', 'measure', 'barrier'
        ]

        def create_observation(circuit):
            """Create observation vector from circuit using public methods."""
            depth = circuit.depth()
            size = circuit.size()
            num_qubits = circuit.num_qubits

            # Count gates by type
            ops = {}
            for instruction in circuit.data:
                gate_name = instruction.operation.name
                ops[gate_name] = ops.get(gate_name, 0) + 1

            # Create observation vector matching environment format
            obs_features = [depth, size, num_qubits] + [
                ops.get(gate, 0) for gate in FIXED_GATE_SET
            ]
            return np.array(obs_features, dtype=np.float32)

        # Create initial observation for our injected circuit
        obs = create_observation(env_direct.circuit)

        # Run the trained PPO agent to optimize the circuit
        terminated = truncated = False
        step_count = 0
        max_steps = 20  # Same as training environment

        print(f"🤖 Starting RL optimization: Initial depth={initial_depth}, "
              f"size={initial_size}")

        while not terminated and not truncated and step_count < max_steps:
            # Get agent's action using the trained policy (deterministic)
            action, _states = model.predict(obs, deterministic=True)
            
            # Ensure action is an integer (sometimes predict returns array)
            if hasattr(action, 'item'):
                action = action.item()

            # Apply the action and get new state
            obs, reward, terminated, truncated, info = env_direct.step(action)
            step_count += 1

            current_depth = env_direct.circuit.depth()
            current_size = env_direct.circuit.size()

            print(f"  Step {step_count}: Action {action}, "
                  f"Depth {env_direct.prev_depth}→{current_depth}, "
                  f"Size {env_direct.prev_size}→{current_size}, "
                  f"Reward {reward:.2f}")

            # Update observation for next step
            obs = create_observation(env_direct.circuit)

            # Early termination if significant improvement is achieved
            if reward > 10:  # High reward threshold
                print("  🎯 High reward achieved, continuing optimization...")

        print(f"  ✅ Optimization completed in {step_count} steps")

        # Extract optimized circuit and metrics
        final_circuit = env_direct.circuit
        final_qasm = qasm2_dumps(final_circuit)
        final_depth = final_circuit.depth()
        final_size = final_circuit.size()
        total_steps = step_count

        # Calculate percentage improvements
        depth_reduction_percent = (
            ((initial_depth - final_depth) / initial_depth * 100)
            if initial_depth > 0 else 0.0
        )
        gate_reduction_percent = (
            ((initial_size - final_size) / initial_size * 100)
            if initial_size > 0 else 0.0
        )

        return OptimizationResponse(
            initial_qasm=initial_qasm,
            optimized_qasm=final_qasm,
            initial_depth=initial_depth,
            initial_size=initial_size,
            optimized_depth=final_depth,
            optimized_size=final_size,
            depth_reduction_percent=depth_reduction_percent,
            gate_reduction_percent=gate_reduction_percent,
            total_steps=total_steps
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for deployment monitoring.

    Returns:
        dict: Application health status and model availability
    """
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "frontend_available": Path(HTML_FILE).exists()
    }


@app.get("/api/status")
async def api_status() -> Dict[str, Any]:
    """
    API status endpoint with detailed information.

    Returns:
        dict: Detailed API and model status
    """
    return {
        "api_version": "1.0.0",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "max_qubits": 15 if dummy_env else None,
        "max_steps": dummy_env.max_steps if dummy_env else None,
        "available_endpoints": [
            "GET /",
            "POST /api/optimize", 
            "GET /api/status",
            "GET /health"
        ]
    }

# === APPLICATION STARTUP ===

if __name__ == "__main__":
    print("Starting RL Quantum Circuit Compiler")
    print("=" * 50)
    print(f"Frontend: {HTML_FILE}")
    print(f"Model: {MODEL_PATH} {'OK' if model_loaded else 'ERROR'}")
    print("Server: http://0.0.0.0:8000")
    print("Health Check: http://0.0.0.0:8000/health")
    print("=" * 50)

    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,  # Changed from 7860 to 8000 for API convention
        log_level="info"
    )
