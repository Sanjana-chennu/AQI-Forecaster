# Stage 1: Base Image
# Start from an official, lightweight Python 3.10 image.
FROM python:3.10-slim

# Stage 2: Set Working Directory
# Create a directory inside the container where our code will live.
WORKDIR /app

# Stage 3: Install Dependencies
# Copy ONLY the requirements file first. This is a Docker optimization.
# If the requirements don't change, Docker can reuse the cached layer, making future builds much faster.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Copy Application Code
# Now, copy all of your application's modules and scripts into the /app directory.
COPY core/ core/
COPY bot/ bot/
COPY main.py .


# Stage 5: Expose Port
# Tell Docker that the application inside this container will listen on port 8000.
EXPOSE 8000

# Stage 6: Run Command
# The command to execute when the container starts. This starts the Uvicorn server.
# The host '0.0.0.0' is crucial for making it accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]