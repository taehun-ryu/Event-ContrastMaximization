#!/usr/bin/env bash
set -euo pipefail

# Basic configuration
IMAGE_NAME="ibec3:latest"
CONTAINER_NAME="ibec3"

# Mount the current host directory into the container
HOST_PROJECT_DIR="$(pwd)"
CONTAINER_PROJECT_DIR="/IBEC3"

# Enable NVIDIA GPU support (comment out to run in CPU-only mode)
USE_GPU=true

# Construct the docker run command
CMD=(docker run -it --rm \
  --name "$CONTAINER_NAME" \
  --net=host \
  --ipc=host \
  --privileged \
  --volume="$HOST_PROJECT_DIR:$CONTAINER_PROJECT_DIR" \
  --volume="${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority" \
  --workdir="$CONTAINER_PROJECT_DIR" \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"
)

# Add GPU flag if enabled
if [ "$USE_GPU" = true ]; then
  CMD+=(--gpus all)
fi

# Add the image name to the command
CMD+=("$IMAGE_NAME")

# Execute the container
"${CMD[@]}"

# Usage
# chmod +x run_ibec3.sh
# ./run_ibec3.sh
