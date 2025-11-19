# --- STAGE 1: 'Builder' ---
# Use a Python 3.10 slim image (Bookworm is excellent for small size)
FROM python:3.10-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# 1. Copy only the requirements file
COPY requirements.txt .

# 2. Install dependencies with pip warning suppressed
# The Qiskit stack is often large, but this slimming process helps.
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt 

# --- STAGE 2: 'Final' ---
# Start again from the same clean base image
FROM python:3.10-slim-bookworm

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

# 6. Expose the FastAPI/Uvicorn port
EXPOSE 8000

# 7. Health Check (Using the dedicated API endpoint)
# We use wget (available in slim image) to hit the /health endpoint we defined in app.py.
# This confirms the Python application has loaded the PPO model and is running successfully.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# 8. Command to run the application (Uvicorn server)
# The deployment service (Render/Railway) will automatically substitute '8000' with its dynamic $PORT.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]