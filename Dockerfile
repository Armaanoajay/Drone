# Use the official Python image as the base image
FROM python:3.11-slim

# Set environment variables to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt ./

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Copy the rest of the application code to the container
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 5000
ENV FLASK_APP=app.py
# Define the command to run the Flask app
ENTRYPOINT ["flask", "run", "--host=0.0.0.0", "--port=5000"]
