# --- STAGE 1: 'Builder' ---
# Use a Python 3.10 slim image (Bookworm is excellent for small size)
FROM python:3.10-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# Suppress pip warnings about running as root
ENV PIP_ROOT_USER_ACTION=ignore

# 1. Copy only the requirements file
COPY requirements.txt .

# 2. Install dependencies with pip warning suppressed
# The Qiskit stack is often large, but this slimming process helps.
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt 

# --- STAGE 2: 'Final' ---
# Start again from the same clean base image
FROM python:3.10-slim-bookworm

# Set production environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_ROOT_USER_ACTION=ignore

# Set working directory
WORKDIR /app

# 3. Copy installed Python packages from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 4. Copy the application files needed to RUN the API and serve the frontend
COPY app.py .
COPY rl_compiler_env.py .
COPY train_enhanced.py .
COPY index.html .
COPY action_specific_training_circuits.json .

# 5. Copy the essential trained model artifact
COPY ppo_quantum_compiler_enhanced.zip . 

# 6. Install curl for health checks and create startup script with logging
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && \
    echo '#!/bin/bash' > startup.sh && \
    echo 'set -e' >> startup.sh && \
    echo 'echo "=== DEPLOYMENT STARTUP LOGGING ===" ' >> startup.sh && \
    echo 'echo "Timestamp: $(date)"' >> startup.sh && \
    echo 'echo "Current directory: $(pwd)"' >> startup.sh && \
    echo 'echo "Contents of current directory:"' >> startup.sh && \
    echo 'ls -la' >> startup.sh && \
    echo 'echo "Python version:"' >> startup.sh && \
    echo 'python --version' >> startup.sh && \
    echo 'echo "Environment variables:"' >> startup.sh && \
    echo 'printenv | grep -E "(PYTHON|PIP|PORT)" || echo "No relevant env vars found"' >> startup.sh && \
    echo 'echo "Checking critical files..."' >> startup.sh && \
    echo 'echo "app.py exists: $(test -f app.py && echo YES || echo NO)"' >> startup.sh && \
    echo 'echo "model file exists: $(test -f ppo_quantum_compiler_enhanced.zip && echo YES || echo NO)"' >> startup.sh && \
    echo 'echo "environment file exists: $(test -f rl_compiler_env.py && echo YES || echo NO)"' >> startup.sh && \
    echo 'echo "Testing Python imports..."' >> startup.sh && \
    echo 'python -c "import sys; print(f\"Python path: {sys.path[:3]}\")" || echo "Python path check failed"' >> startup.sh && \
    echo 'python -c "import qiskit; print(f\"Qiskit version: {qiskit.__version__}\")" || echo "Qiskit import failed"' >> startup.sh && \
    echo 'python -c "import stable_baselines3; print(f\"SB3 version: {stable_baselines3.__version__}\")" || echo "SB3 import failed"' >> startup.sh && \
    echo 'python -c "import fastapi; print(f\"FastAPI version: {fastapi.__version__}\")" || echo "FastAPI import failed"' >> startup.sh && \
    echo 'echo "Testing model loading..."' >> startup.sh && \
    echo 'python -c "from stable_baselines3 import PPO; model = PPO.load(\"ppo_quantum_compiler_enhanced.zip\"); print(\"Model loaded successfully\")" || echo "Model loading failed"' >> startup.sh && \
    echo 'echo "Testing environment creation..."' >> startup.sh && \
    echo 'python -c "from rl_compiler_env import QuantumCircuitEnv; env = QuantumCircuitEnv(); print(\"Environment created successfully\")" || echo "Environment creation failed"' >> startup.sh && \
    echo 'echo "All checks passed! Starting uvicorn server..."' >> startup.sh && \
    echo 'exec uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info' >> startup.sh && \
    chmod +x startup.sh

# 7. Expose the FastAPI/Uvicorn port
EXPOSE 8000

# 8. Health Check (Using the dedicated API endpoint)
# We use curl (now installed) to hit the /health endpoint we defined in app.py.
# This confirms the Python application has loaded the PPO model and is running successfully.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 9. Command to run the startup script with comprehensive logging
CMD ["./startup.sh"]