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
CONTAINER_NAME="sglang_xutingz"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

log_status() {
    echo -e "${CYAN}[STATUS]${NC} $1"
}

# 检查单个节点的容器状态
check_container_on_node() {
    local ip=$1
    local node_name=${NODE_NAMES[$ip]}
    
    # SSH命令
    ssh_cmd="ssh -o ConnectTimeout=$SSH_TIMEOUT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSH_USER@$ip"
    
    # 首先测试SSH连接
    if ! $ssh_cmd "echo 'test'" >/dev/null 2>&1; then
        log_error "SSH connection failed to $ip ($node_name)"
        return 2  # 连接失败
    fi
    
    # 检查Docker是否安装
    if ! $ssh_cmd "which docker" >/dev/null 2>&1; then
        log_error "Docker not found on $ip ($node_name)"
        return 3  # Docker未安装
    fi
    
    # 检查容器是否存在且运行
    local container_status
    container_status=$($ssh_cmd "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep '^${CONTAINER_NAME}'" 2>/dev/null)
    
    if [ -n "$container_status" ]; then
        log_success "Container running on $ip ($node_name)"
        echo "  $container_status"
        return 0  # 容器运行中
    fi
    
    # 检查容器是否存在但未运行
    local stopped_container
    stopped_container=$($ssh_cmd "docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep '^${CONTAINER_NAME}'" 2>/dev/null)
    
    if [ -n "$stopped_container" ]; then
        log_warning "Container exists but not running on $ip ($node_name)"
        echo "  $stopped_container"
        return 1  # 容器存在但未运行
    fi
    
    # 容器不存在
    log_error "Container not found on $ip ($node_name)"
    return 4  # 容器不存在
}

# 并行检查容器状态的包装函数
check_container_parallel() {
    local ip=$1
    local result_file="/tmp/container_check_${ip//\./_}.result"
    
    # 检查容器状态并将结果写入临时文件
    local status_code
    {
        check_container_on_node "$ip"
        status_code=$?
    } > "/tmp/container_check_${ip//\./_}.output" 2>&1
    
    echo "$status_code:$ip" > "$result_file"
}

# 主检查函数 - 并行版本
main_check() {
    log_info "Checking Docker container '$CONTAINER_NAME' status on ${#NODES[@]} nodes..."
    echo
    
    local pids=()
    local running_count=0
    local stopped_count=0
    local missing_count=0
    local ssh_failed_count=0
    local docker_missing_count=0
    
    local running_nodes=()
    local stopped_nodes=()
    local missing_nodes=()
    local ssh_failed_nodes=()
    local docker_missing_nodes=()
    
    # 清理之前的结果文件
    rm -f /tmp/container_check_*.result /tmp/container_check_*.output 2>/dev/null
    
    # 并行启动所有节点的检查
    for ip in "${NODES[@]}"; do
        check_container_parallel "$ip" &
        pids+=($!)
    done
    
    log_info "All check processes started. Waiting for completion..."
    echo
    
    # 等待所有后台进程完成
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    # 显示详细输出
    for ip in "${NODES[@]}"; do
        local output_file="/tmp/container_check_${ip//\./_}.output"
        if [ -f "$output_file" ]; then
            cat "$output_file"
        fi
    done
    
    echo
    
    # 收集结果
    for ip in "${NODES[@]}"; do
        local result_file="/tmp/container_check_${ip//\./_}.result"
        local node_name=${NODE_NAMES[$ip]}
        
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            local status_code=${result%%:*}
            
            case $status_code in
                0)  # 容器运行中
                    ((running_count++))
                    running_nodes+=("$ip ($node_name)")
                    ;;
                1)  # 容器存在但未运行
                    ((stopped_count++))
                    stopped_nodes+=("$ip ($node_name)")
                    ;;
                2)  # SSH连接失败
                    ((ssh_failed_count++))
                    ssh_failed_nodes+=("$ip ($node_name)")
                    ;;
                3)  # Docker未安装
                    ((docker_missing_count++))
                    docker_missing_nodes+=("$ip ($node_name)")
                    ;;
                4)  # 容器不存在
                    ((missing_count++))
                    missing_nodes+=("$ip ($node_name)")
                    ;;
            esac
            rm -f "$result_file"
        else
            missing_nodes+=("$ip ($node_name) - No result file")
            ((missing_count++))
        fi
    done
    
    # 清理输出文件
    rm -f /tmp/container_check_*.output 2>/dev/null
    
    # 输出总结
    echo "=========================================="
    log_info "Container Status Summary:"
    echo
    
    if [ $running_count -gt 0 ]; then
        log_success "Nodes with running containers: $running_count"
        for node in "${running_nodes[@]}"; do
            echo "  ✓ $node"
        done
        echo
    fi
    
    if [ $stopped_count -gt 0 ]; then
        log_warning "Nodes with stopped containers: $stopped_count"
        for node in "${stopped_nodes[@]}"; do
            echo "  ⏸ $node"
        done
        echo
    fi
    
    if [ $missing_count -gt 0 ]; then
        log_error "Nodes missing containers: $missing_count"
        for node in "${missing_nodes[@]}"; do
            echo "  ✗ $node"
        done
        echo
    fi
    
    if [ $ssh_failed_count -gt 0 ]; then
        log_error "Nodes with SSH connection issues: $ssh_failed_count"
        for node in "${ssh_failed_nodes[@]}"; do
            echo "  🔌 $node"
        done
        echo
    fi
    
    if [ $docker_missing_count -gt 0 ]; then
        log_error "Nodes without Docker: $docker_missing_count"
        for node in "${docker_missing_nodes[@]}"; do
            echo "  🐳 $node"
        done
        echo
    fi
    
    # 统计信息
    echo "=========================================="
    log_status "Statistics:"
    echo "  Total nodes: ${#NODES[@]}"
    echo "  Running containers: $running_count"
    echo "  Stopped containers: $stopped_count"
    echo "  Missing containers: $missing_count"
    echo "  SSH failures: $ssh_failed_count"
    echo "  Docker not installed: $docker_missing_count"
    
    # 返回适当的退出码
    if [ $missing_count -gt 0 ] || [ $ssh_failed_count -gt 0 ] || [ $docker_missing_count -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# 串行检查函数
main_check_serial() {
    log_info "Checking Docker container '$CONTAINER_NAME' status on ${#NODES[@]} nodes (serial mode)..."
    echo
    
    local running_count=0
    local stopped_count=0
    local missing_count=0
    local ssh_failed_count=0
    local docker_missing_count=0
    
    local running_nodes=()
    local stopped_nodes=()
    local missing_nodes=()
    local ssh_failed_nodes=()
    local docker_missing_nodes=()
    
    # 遍历所有节点
    for ip in "${NODES[@]}"; do
        local node_name=${NODE_NAMES[$ip]}
        local status_code
        
        check_container_on_node "$ip"
        status_code=$?
        
        case $status_code in
            0)  # 容器运行中
                ((running_count++))
                running_nodes+=("$ip ($node_name)")
                ;;
            1)  # 容器存在但未运行
                ((stopped_count++))
                stopped_nodes+=("$ip ($node_name)")
                ;;
            2)  # SSH连接失败
                ((ssh_failed_count++))
                ssh_failed_nodes+=("$ip ($node_name)")
                ;;
            3)  # Docker未安装
                ((docker_missing_count++))
                docker_missing_nodes+=("$ip ($node_name)")
                ;;
            4)  # 容器不存在
                ((missing_count++))
                missing_nodes+=("$ip ($node_name)")
                ;;
        esac
        echo
    done
    
    # 输出总结（与并行版本相同的格式）
    echo "=========================================="
    log_info "Container Status Summary:"
    echo
    
    if [ $running_count -gt 0 ]; then
        log_success "Nodes with running containers: $running_count"
        for node in "${running_nodes[@]}"; do
            echo "  ✓ $node"
        done
        echo
    fi
    
    if [ $stopped_count -gt 0 ]; then
        log_warning "Nodes with stopped containers: $stopped_count"
        for node in "${stopped_nodes[@]}"; do
            echo "  ⏸ $node"
        done
        echo
    fi
    
    if [ $missing_count -gt 0 ]; then
        log_error "Nodes missing containers: $missing_count"
        for node in "${missing_nodes[@]}"; do
            echo "  ✗ $node"
        done
        echo
    fi
    
    if [ $ssh_failed_count -gt 0 ]; then
        log_error "Nodes with SSH connection issues: $ssh_failed_count"
        for node in "${ssh_failed_nodes[@]}"; do
            echo "  🔌 $node"
        done
        echo
    fi
    
    if [ $docker_missing_count -gt 0 ]; then
        log_error "Nodes without Docker: $docker_missing_count"
        for node in "${docker_missing_nodes[@]}"; do
            echo "  🐳 $node"
        done
        echo
    fi
    
    # 统计信息
    echo "=========================================="
    log_status "Statistics:"
    echo "  Total nodes: ${#NODES[@]}"
    echo "  Running containers: $running_count"
    echo "  Stopped containers: $stopped_count"
    echo "  Missing containers: $missing_count"
    echo "  SSH failures: $ssh_failed_count"
    echo "  Docker not installed: $docker_missing_count"
    
    # 返回适当的退出码
    if [ $missing_count -gt 0 ] || [ $ssh_failed_count -gt 0 ] || [ $docker_missing_count -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# 显示帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Check Docker container '$CONTAINER_NAME' status across all nodes."
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -s, --serial   Run in serial mode (one node at a time)"
    echo "  -p, --parallel Run in parallel mode (all nodes simultaneously) [default]"
    echo "  -n, --node IP  Check specific node only (e.g., -n 172.16.1.249)"
    echo "  -l, --list     List all available nodes"
    echo
    echo "Examples:"
    echo "  $0                    # Check all nodes (parallel mode)"
    echo "  $0 --serial           # Check all nodes (serial mode)"
    echo "  $0 -n 172.16.1.249   # Check specific node (n0)"
    echo "  $0 --list             # List all nodes"
}

# 列出所有节点
list_nodes() {
    echo "Available nodes:"
    for ip in "${NODES[@]}"; do
        echo "  $ip (${NODE_NAMES[$ip]})"
    done
}

# 检查特定节点
check_specific_node() {
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
    
    log_info "Checking Docker container on specific node: $target_ip (${NODE_NAMES[$target_ip]})"
    echo
    check_container_on_node "$target_ip"
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
            check_specific_node "$2"
            exit $?
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
    main_check_serial
else
    main_check
fi 
