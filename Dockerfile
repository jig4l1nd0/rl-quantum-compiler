# --- STAGE 1: 'Builder' ---
# Use a Python 3.10 slim image (Bookworm is excellent for small size)
FROM python:3.10-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# 1. Copy only the requirements file
COPY requirements.txt .

# 2. Install dependencies (still as root in builder stage for permissions)
# The Qiskit stack is often large, but this slimming process helps.
RUN pip install --no-cache-dir -r requirements.txt 

# --- STAGE 2: 'Final' ---
# Start again from the same clean base image
FROM python:3.10-slim-bookworm

# Create a non-root user for security and to avoid pip warnings
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set working directory and change ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# 3. Copy installed Python packages from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 4. Copy the application files needed to RUN the API and serve the frontend
# We need all Python source files and the static HTML file.
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser rl_compiler_env.py .
COPY --chown=appuser:appuser train_enhanced.py .
COPY --chown=appuser:appuser index.html .
COPY --chown=appuser:appuser action_specific_training_circuits.json .

# 5. Copy the essential trained model artifact
# This is required for the application to run the PPO inference
COPY --chown=appuser:appuser ppo_quantum_compiler_enhanced.zip . 

# Switch to non-root user
USER appuser 

# 6. Expose the FastAPI/Uvicorn port
# This is required for the deployment service (Render/Railway) to proxy traffic.
EXPOSE 8000

# 7. Health Check (Using the dedicated API endpoint)
# We use wget (available in slim image) to hit the /health endpoint we defined in app.py.
# This confirms the Python application has loaded the PPO model and is running successfully.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# 8. Command to run the application (Uvicorn server)
# The deployment service (Render/Railway) will automatically substitute '8000' with its dynamic $PORT.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]