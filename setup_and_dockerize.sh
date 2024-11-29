#!/bin/bash

# Variables (modify these as per your setup)
REPO_URL="https://github.com/Rahul-404/Chat_With_Multiple_PDF_Using_LangChain_And_LLMs.git"   # GitHub repo URL
PROJECT_DIR="Chat_With_Multiple_PDF_Using_LangChain_And_LLMs"                                 # Project directory name
DOCKER_IMAGE_NAME="gemini-app"                                                                # Docker image name
DOCKER_CONTAINER_NAME="your-container-name"                                                   # Docker container name
AWS_REGION="us-east-1"                                                                        # AWS region

# Update system and install dependencies
echo "Updating system packages..."
sudo yum update -y

echo "Installing Docker and Git..."
# Install Docker and Git (if not installed)
sudo yum install -y docker git

# Start Docker service
echo "Starting Docker service..."
sudo service docker start

# Allow ec2-user to run docker without sudo
echo "Allowing ec2-user to use Docker without sudo..."
sudo usermod -aG docker ec2-user
newgrp docker

# Clone the repository
echo "Cloning the repository..."
cd /home/ec2-user
git clone $REPO_URL
cd $PROJECT_DIR

# If your project doesn't have a Dockerfile, create a simple one
# Optional: Create a Dockerfile if your project doesn't already have one
# if [ ! -f Dockerfile ]; then
    # echo "Creating a basic Dockerfile for the project..."
    # cat > Dockerfile <<EOL
# Use an official Node.js runtime as the base image
# FROM node:16

# Set the working directory in the container
# WORKDIR /app

# Copy package.json and package-lock.json (for Node.js projects)
# COPY package*.json ./

# Install dependencies
# RUN npm install

# Copy the rest of the project files
# COPY . .

# Expose the port the app runs on (adjust this if needed)
# EXPOSE 3000

# Run the application
# CMD ["npm", "start"]
# EOL
# fi

# Build the Docker image
echo "Building Docker image..."
docker build -t $DOCKER_IMAGE_NAME .

# Run the Docker container
echo "Running Docker container..."
docker run -d -p 3000:3000 --name $DOCKER_CONTAINER_NAME $DOCKER_IMAGE_NAME

echo "Docker container is now running. You can access your application at http://<EC2_PUBLIC_IP>:3000"

# Output public IP of EC2 instance
EC2_PUBLIC_IP=$(curl http://checkip.amazonaws.com)
echo "EC2 instance public IP: $EC2_PUBLIC_IP"
