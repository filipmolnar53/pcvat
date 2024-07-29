#!/bin/bash
docker compose up -d

if ! command -v nuctl &> /dev/null
then
    echo "nuctl not found, downloading..."
    wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64
    sudo chmod +x nuctl-1.13.0-linux-amd64
    sudo ln -sf $(pwd)/nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl
else
    echo "nuctl is already installed"
fi

docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'


serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam/
