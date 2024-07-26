#!/bin/bash

# Step 1: Start Docker Compose services in detached mode
docker compose up -d

# Step 2: Download nuctl
if ! command -v nuctl &> /dev/null
then
    echo "nuctl not found, downloading..."
    wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64
    sudo chmod +x nuctl-1.13.0-linux-amd64
    sudo ln -sf $(pwd)/nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl
else
    echo "nuctl is already installed"
fi

# Step 3: Make nuctl executable
sudo chmod +x nuctl-1.13.0-linux-amd64

# Step 4: Create a symbolic link to nuctl in /usr/local/bin
sudo ln -sf $(pwd)/nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl

# Step 5: Run the deploy_gpu.sh script
serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam/
