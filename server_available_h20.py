import subprocess
import concurrent.futures
import time

# --- 配置 ---
PRIVATE_IPS = [
    "172.16.1.249", "172.16.1.9", "172.16.1.1", "172.16.1.250",
    "172.16.1.251", "172.16.1.6", "172.16.1.10", "172.16.1.252",
    "172.16.1.8", "172.16.1.5", "172.16.1.12", "172.16.1.14",
    "172.16.1.3", "172.16.1.13", "172.16.1.11", "172.16.1.15",
    "172.16.1.4"
]

# IP到节点别名的映射
IP_TO_NODE = {
    "172.16.1.249": "n0", "172.16.1.9": "n1", "172.16.1.1": "n2", "172.16.1.250": "n3",
    "172.16.1.251": "n4", "172.16.1.6": "n5", "172.16.1.10": "n6", "172.16.1.252": "n7",
    "172.16.1.8": "n8", "172.16.1.5": "n9", "172.16.1.12": "n10", "172.16.1.14": "n11",
    "172.16.1.3": "n12", "172.16.1.13": "n13", "172.16.1.11": "n14", "172.16.1.15": "n15",
    "172.16.1.4": "n16"
}

SSH_USER = "root"
SSH_TIMEOUT = 5 # SSH 连接超时时间（秒）
CMD_TIMEOUT = 10 # 远程命令执行超时时间（秒）
MAX_WORKERS = 10 # 同时检查的最大节点数 (根据本地能力和网络调整)
# --- 配置结束 ---

def check_gpu_availability(ip):
    """
    通过 SSH 连接到节点并检查 GPU 是否空闲。
    如果 GPU 空闲（没有运行中的进程），返回 IP 地址。
    否则返回 None。
    """
    ssh_command = [
        "ssh",
        "-q",  # 安静模式，禁止大多数警告和诊断消息
        f"-o ConnectTimeout={SSH_TIMEOUT}",
        "-o StrictHostKeyChecking=no",  # 首次连接时自动接受主机密钥 (不安全，但方便脚本)
        "-o UserKnownHostsFile=/dev/null", # 不将主机密钥添加到 known_hosts (不安全)
        f"{SSH_USER}@{ip}",
        "nvidia-smi" # 要在远程主机上执行的命令
    ]
    print(f"[{time.strftime('%H:%M:%S')}] Checking {ip}...")
    try:
        # 执行 SSH 命令
        result = subprocess.run(
            ssh_command,
            capture_output=True, # 捕获标准输出和标准错误
            text=True,           # 将输出解码为文本
            check=True,          # 如果命令返回非零退出码，则引发 CalledProcessError
            timeout=CMD_TIMEOUT  # 为整个命令执行设置超时
        )

        # 检查 nvidia-smi 输出中是否包含 "No running processes found"
        # 这是判断 GPU 空闲的一个简单但常用的方法
        if "No running processes found" in result.stdout:
            node_name = IP_TO_NODE.get(ip, ip)
            print(f"[{time.strftime('%H:%M:%S')}] -> GPUs FREE on {ip} ({node_name})")
            return ip
        else:
            # 解析是否有进程列表 (更精确的检查，但稍微复杂)
            # 简单的做法是：如果没有明确的"No running processes"，就认为它可能在忙
            node_name = IP_TO_NODE.get(ip, ip)
            print(f"[{time.strftime('%H:%M:%S')}] -> GPUs BUSY or Unknown on {ip} ({node_name})")
            return None

    except subprocess.CalledProcessError as e:
        # 命令执行了，但返回了错误码 (例如 nvidia-smi 不存在)
        print(f"[{time.strftime('%H:%M:%S')}] -> ERROR checking {ip}: Command failed. Stderr: {e.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        # 命令执行超时
        print(f"[{time.strftime('%H:%M:%S')}] -> ERROR checking {ip}: Command timed out after {CMD_TIMEOUT}s.")
        return None
    except Exception as e:
        # 其他错误 (例如 SSH 连接失败、权限问题等)
        print(f"[{time.strftime('%H:%M:%S')}] -> ERROR checking {ip}: {type(e).__name__} - {e}")
        return None

if __name__ == "__main__":
    print(f"Starting GPU availability check for {len(PRIVATE_IPS)} nodes...")
    free_nodes = []

    # 使用线程池并行执行检查
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 创建 future 任务
        future_to_ip = {executor.submit(check_gpu_availability, ip): ip for ip in PRIVATE_IPS}

        # 等待任务完成并收集结果
        for future in concurrent.futures.as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                result_ip = future.result() # 获取 check_gpu_availability 的返回值
                if result_ip: # 如果返回了 IP 地址，说明是空闲的
                    free_nodes.append(result_ip)
            except Exception as exc:
                # 一般来说，check_gpu_availability 内部会处理异常
                # 但以防万一，这里也加一个捕获
                print(f"[{time.strftime('%H:%M:%S')}] -> UNEXPECTED ERROR processing result for {ip}: {exc}")

    print("\n--- Check Complete ---")
    if free_nodes:
        print("Nodes with potentially free GPUs:")
        free_nodes.sort() # 对结果进行排序
        for node_ip in free_nodes:
            node_name = IP_TO_NODE.get(node_ip, node_ip)
            print(f"{node_ip} ({node_name})")
    else:
        print("No nodes found with apparently free GPUs based on 'nvidia-smi' check.")

