#!/bin/bash

# 节点IP列表
NODES=(
    "172.16.1.249"  # n0
    "172.16.1.9"    # n1
    "172.16.1.1"    # n2
    "172.16.1.250"  # n3
    "172.16.1.251"  # n4
    "172.16.1.6"    # n5
    "172.16.1.10"   # n6
    "172.16.1.252"  # n7
    "172.16.1.8"    # n8
    "172.16.1.5"    # n9
    "172.16.1.12"   # n10
    "172.16.1.14"   # n11
    "172.16.1.3"    # n12
    "172.16.1.13"   # n13
    "172.16.1.11"   # n14
    "172.16.1.15"   # n15
    "172.16.1.4"    # n16
)

# 节点别名映射
declare -A NODE_NAMES
NODE_NAMES["172.16.1.249"]="n0"
NODE_NAMES["172.16.1.9"]="n1"
NODE_NAMES["172.16.1.1"]="n2"
NODE_NAMES["172.16.1.250"]="n3"
NODE_NAMES["172.16.1.251"]="n4"
NODE_NAMES["172.16.1.6"]="n5"
NODE_NAMES["172.16.1.10"]="n6"
NODE_NAMES["172.16.1.252"]="n7"
NODE_NAMES["172.16.1.8"]="n8"
NODE_NAMES["172.16.1.5"]="n9"
NODE_NAMES["172.16.1.12"]="n10"
NODE_NAMES["172.16.1.14"]="n11"
NODE_NAMES["172.16.1.3"]="n12"
NODE_NAMES["172.16.1.13"]="n13"
NODE_NAMES["172.16.1.11"]="n14"
NODE_NAMES["172.16.1.15"]="n15"
NODE_NAMES["172.16.1.4"]="n16"

# SSH配置
SSH_USER="root"
SSH_TIMEOUT=10

# Docker命令
DOCKER_CMD="docker run -itd --shm-size 32g --gpus all -v /mnt/.cache:/root/.cache --ipc=host --network=host --privileged --name sglang_xutingz registry.cn-beijing.aliyuncs.com/eai_beijing/sglang:dev-deepep /bin/zsh"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 在单个节点上运行Docker命令
run_docker_on_node() {
    local ip=$1
    local node_name=${NODE_NAMES[$ip]}
    
    log_info "Connecting to $ip ($node_name)..."
    
    # SSH命令
    ssh_cmd="ssh -o ConnectTimeout=$SSH_TIMEOUT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSH_USER@$ip"
    
    # 首先测试SSH连接
    if ! $ssh_cmd "echo 'SSH connection test'" >/dev/null 2>&1; then
        log_error "SSH connection failed to $ip ($node_name)"
        return 1
    fi
    
    # 检查Docker是否安装
    if ! $ssh_cmd "which docker" >/dev/null 2>&1; then
        log_error "Docker not found on $ip ($node_name)"
        return 1
    fi
    
    # 检查并清理已存在的同名容器
    local container_name="sglang_xutingz"
    if $ssh_cmd "docker ps -a --format '{{.Names}}' | grep -q '^${container_name}$'" 2>/dev/null; then
        log_warning "Container '$container_name' already exists on $ip ($node_name), removing..."
        $ssh_cmd "docker rm -f $container_name" >/dev/null 2>&1
    fi
    
    # 执行Docker命令并捕获错误
    local output
    local exit_code
    output=$($ssh_cmd "$DOCKER_CMD" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Docker container started successfully on $ip ($node_name)"
        echo "  Container ID: $(echo "$output" | tail -1)"
        return 0
    else
        log_error "Failed to start Docker container on $ip ($node_name)"
        echo "  Error: $output"
        return 1
    fi
}

# 并行运行Docker命令的包装函数
run_docker_parallel() {
    local ip=$1
    local result_file="/tmp/docker_deploy_${ip//\./_}.result"
    
    # 运行Docker命令并将结果写入临时文件
    if run_docker_on_node "$ip"; then
        echo "SUCCESS:$ip" > "$result_file"
    else
        echo "FAILED:$ip" > "$result_file"
    fi
}

# 主函数 - 并行版本
main() {
    log_info "Starting parallel Docker deployment on ${#NODES[@]} nodes..."
    echo
    
    local pids=()
    local success_count=0
    local failed_nodes=()
    
    # 清理之前的结果文件
    rm -f /tmp/docker_deploy_*.result 2>/dev/null
    
    # 并行启动所有节点的部署
    for ip in "${NODES[@]}"; do
        run_docker_parallel "$ip" &
        pids+=($!)
        log_info "Started deployment process for $ip (${NODE_NAMES[$ip]}) - PID: $!"
    done
    
    echo
    log_info "All deployment processes started. Waiting for completion..."
    echo
    
    # 等待所有后台进程完成
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    # 收集结果
    for ip in "${NODES[@]}"; do
        local result_file="/tmp/docker_deploy_${ip//\./_}.result"
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            if [[ "$result" == "SUCCESS:"* ]]; then
                ((success_count++))
            else
                failed_nodes+=("$ip (${NODE_NAMES[$ip]})")
            fi
            rm -f "$result_file"
        else
            failed_nodes+=("$ip (${NODE_NAMES[$ip]}) - No result file")
        fi
    done
    
    # 输出总结
    echo "=================================="
    log_info "Parallel Deployment Summary:"
    log_success "Successfully deployed on $success_count/${#NODES[@]} nodes"
    
    if [ ${#failed_nodes[@]} -gt 0 ]; then
        log_warning "Failed nodes:"
        for node in "${failed_nodes[@]}"; do
            echo "  - $node"
        done
    fi
    
    echo
    log_info "Docker command used:"
    echo "$DOCKER_CMD"
}

# 串行主函数（原版本）
main_serial() {
    log_info "Starting serial Docker deployment on ${#NODES[@]} nodes..."
    echo
    
    local success_count=0
    local failed_nodes=()
    
    # 遍历所有节点
    for ip in "${NODES[@]}"; do
        if run_docker_on_node "$ip"; then
            ((success_count++))
        else
            failed_nodes+=("$ip (${NODE_NAMES[$ip]})")
        fi
        echo
    done
    
    # 输出总结
    echo "=================================="
    log_info "Serial Deployment Summary:"
    log_success "Successfully deployed on $success_count/${#NODES[@]} nodes"
    
    if [ ${#failed_nodes[@]} -gt 0 ]; then
        log_warning "Failed nodes:"
        for node in "${failed_nodes[@]}"; do
            echo "  - $node"
        done
    fi
    
    echo
    log_info "Docker command used:"
    echo "$DOCKER_CMD"
}

# 显示帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -n, --node IP  Run on specific node only (e.g., -n 172.16.1.249)"
    echo "  -l, --list     List all available nodes"
    echo "  -t, --test     Test SSH connectivity to all nodes"
    echo "  -s, --serial   Run in serial mode (one node at a time)"
    echo "  -p, --parallel Run in parallel mode (all nodes simultaneously) [default]"
    echo
    echo "Examples:"
    echo "  $0                    # Run on all nodes (parallel mode)"
    echo "  $0 --serial           # Run on all nodes (serial mode)"
    echo "  $0 -n 172.16.1.249   # Run on specific node (n0)"
    echo "  $0 --list             # List all nodes"
    echo "  $0 --test             # Test connectivity"
}

# 列出所有节点
list_nodes() {
    echo "Available nodes:"
    for ip in "${NODES[@]}"; do
        echo "  $ip (${NODE_NAMES[$ip]})"
    done
}

# 测试所有节点的SSH连接
test_connectivity() {
    log_info "Testing SSH connectivity to all nodes..."
    echo
    
    local reachable_count=0
    local unreachable_nodes=()
    
    for ip in "${NODES[@]}"; do
        local node_name=${NODE_NAMES[$ip]}
        log_info "Testing connection to $ip ($node_name)..."
        
        ssh_cmd="ssh -o ConnectTimeout=$SSH_TIMEOUT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSH_USER@$ip"
        
        if $ssh_cmd "echo 'Connection successful'" >/dev/null 2>&1; then
            log_success "SSH connection successful to $ip ($node_name)"
            ((reachable_count++))
        else
            log_error "SSH connection failed to $ip ($node_name)"
            unreachable_nodes+=("$ip ($node_name)")
        fi
    done
    
    echo
    echo "=================================="
    log_info "Connectivity Test Summary:"
    log_success "Reachable nodes: $reachable_count/${#NODES[@]}"
    
    if [ ${#unreachable_nodes[@]} -gt 0 ]; then
        log_warning "Unreachable nodes:"
        for node in "${unreachable_nodes[@]}"; do
            echo "  - $node"
        done
    fi
}

# 在特定节点上运行
run_on_specific_node() {
    local target_ip=$1
    local found=false
    
    for ip in "${NODES[@]}"; do
        if [ "$ip" = "$target_ip" ]; then
            found=true
            break
        fi
    done
    
    if [ "$found" = false ]; then
        log_error "Node $target_ip not found in the node list"
        echo
        list_nodes
        exit 1
    fi
    
    log_info "Running Docker command on specific node: $target_ip (${NODE_NAMES[$target_ip]})"
    echo
    run_docker_on_node "$target_ip"
}

# 默认使用并行模式
SERIAL_MODE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--list)
            list_nodes
            exit 0
            ;;
        -t|--test)
            test_connectivity
            exit 0
            ;;
        -s|--serial)
            SERIAL_MODE=true
            ;;
        -p|--parallel)
            SERIAL_MODE=false
            ;;
        -n|--node)
            if [ -z "$2" ]; then
                log_error "Node IP required after -n/--node option"
                exit 1
            fi
            run_on_specific_node "$2"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# 根据模式选择运行函数
if [ "$SERIAL_MODE" = true ]; then
    main_serial
else
    main
fi 
