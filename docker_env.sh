#!/bin/bash

# ================= 配置区域 =================
# 容器名字 (保持固定，方便管理)
CONTAINER_NAME="jdiff_work"
# 镜像名字 (你的快照或官方镜像)
IMAGE_NAME="jdiffusion:latest"
# 挂载目录 (默认挂载当前脚本所在目录到容器的 /workspace)
HOST_DIR="$(pwd)"
WORK_DIR="/workspace"
# ===========================================

# 获取操作指令
ACTION=$1

# 帮助信息
show_help() {
    echo "使用方法: ./manage.sh [start|enter|stop|rm]"
    echo "  start : 启动或重启容器 (后台模式)"
    echo "  enter : 进入容器 (打开终端)"
    echo "  stop  : 停止容器 (不删除数据)"
    echo "  rm    : 停止并彻底删除容器"
}

# 1. 启动容器逻辑
start_container() {
    # 检查容器是否已经在运行
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "✅ 容器 [$CONTAINER_NAME] 正在运行中。"
    else
        # 检查容器是否存在但停止了
        if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
            echo "🔄 容器已存在但停止了，正在唤醒..."
            docker start $CONTAINER_NAME
        else
            echo "🚀 正在创建并启动新容器..."
            # 核心启动命令
            docker run -dt \
                --gpus all \
                --name $CONTAINER_NAME \
                -v "$HOST_DIR:$WORK_DIR" \
                $IMAGE_NAME \
                /bin/bash
        fi
        echo "✅ 容器启动成功！"
    fi
}

# 2. 进入容器逻辑
enter_container() {
    # 确保容器在运行
    if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "⚠️ 容器未运行，正在尝试启动..."
        start_container
    fi
    echo "root@container:~$ 进入容器工作环境 (输入 exit 退出)..."
    # docker exec -it $CONTAINER_NAME /bin/bash
    docker exec -it $CONTAINER_NAME /bin/bash -c "exec /bin/bash --rcfile <(echo '. ~/.bashrc; source /root/anaconda3/etc/profile.d/conda.sh; conda activate jdiffusion; cd /workspace')"
}

# 3. 停止容器
stop_container() {
    echo "🛑 正在停止容器..."
    docker stop $CONTAINER_NAME
    echo "✅ 容器已停止。"
}

# 4. 删除容器
remove_container() {
    echo "🗑️ 正在删除容器..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1
    docker rm $CONTAINER_NAME
    echo "✅ 容器已删除 (宿主机代码保留)。"
}

# 主逻辑路由
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