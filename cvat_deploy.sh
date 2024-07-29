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

usage() {
  echo "Usage: $0 [-su] [--create-superuser]"
  exit 1
}

# Parse options
while [[ "$1" != "" ]]; do
  case $1 in
    -su | --create-superuser )
      s_flag=true
      ;;
    -h | --help )
      usage
      ;;
    * )
      echo "Invalid option: $1"
      usage
      ;;
  esac
  shift
done

# Check if the -s or --create-su flag was provided
if [ "$s_flag" = true ]; then
    docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
fi


serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam/
