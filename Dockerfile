# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Create the directory /path/in/container for shared volume
RUN mkdir -p /path/in/container

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for the model and label encoder paths
ENV MODEL_PATH /app/my_multiclass_model.h5
ENV ENCODER_PATH /app/label_encoder.pkl

# Run gunicorn when the container launches
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
