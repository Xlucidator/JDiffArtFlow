#!/bin/bash

# ================= Configuration =================
CONTAINER_NAME="jdiff_work"
IMAGE_NAME="jdiffusion:latest"
HOST_DIR="$(pwd)"
WORK_DIR="/workspace"
# =================================================

# Get action command
ACTION=$1

show_help() {
    echo "Usage: ./manage.sh [start|enter|stop|rm]"
    echo "  start : 启动或重启容器 (后台模式)"
    echo "  enter : 进入容器 (打开终端)"
    echo "  stop  : 停止容器 (不删除数据)"
    echo "  rm    : 停止并彻底删除容器"
}

# 1. Start container logic
start_container() {
    # Check if the container is already running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "✅ Container [$CONTAINER_NAME] is already running."
    else
        # Check if the container exists but is stopped
        if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
            echo "🔄 Container exists but is stopped, waking up..."
            docker start $CONTAINER_NAME
        else
            echo "🚀 Creating and starting a new container..."
            # Core start command
            docker run -dt \
                --gpus all \
                --name $CONTAINER_NAME \
                -v "$HOST_DIR:$WORK_DIR" \
                $IMAGE_NAME \
                /bin/bash
        fi
        echo "✅ Container started successfully!"
    fi
}

# 2. Enter container logic
enter_container() {
    # Ensure the container is running
    if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "⚠️ Container is not running, attempting to start..."
        start_container
    fi
    echo "root@container:~$ Entering container workspace (type exit to quit)..."
    # docker exec -it $CONTAINER_NAME /bin/bash
    docker exec -it $CONTAINER_NAME /bin/bash -c "exec /bin/bash --rcfile <(echo '. ~/.bashrc; source /root/anaconda3/etc/profile.d/conda.sh; conda activate jdiffusion; cd /workspace')"
}

# 3. Stop container
stop_container() {
    echo "🛑 Stopping container..."
    docker stop $CONTAINER_NAME
    echo "✅ Container stopped."
}

# 4. Remove container
remove_container() {
    echo "🗑️ Removing container..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1
    docker rm $CONTAINER_NAME
    echo "✅ Container removed (host code preserved)."
}

# Main logic routing
case "$ACTION" in
    start)
        start_container
        ;;
    enter)
        enter_container
        ;;
    stop)
        stop_container
        ;;
    rm)
        remove_container
        ;;
    *)
        show_help
        ;;
esac