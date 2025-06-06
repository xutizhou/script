import triton
import time

def exp2_upper(num: int) -> int:
    for i in range(2, 31):
        value = pow(2, i)
        if num <= value:
            return value
    return num

def safe_next_power_of_2(n: int) -> int:
    """安全版本的next_power_of_2，正确处理0和负数"""
    if n <= 0:
        return 1  # 对于0或负数，返回1
    return triton.next_power_of_2(n)

def exp2_upper_fixed(num: int) -> int:
    """修复版本的exp2_upper，能处理0"""
    if num <= 0:
        return 1  # 对于0或负数，返回1
    for i in range(1, 31):  # 从i=1开始，包含2^1=2
        value = pow(2, i)
        if num <= value:
            return value
    return num

def test_zero_and_negative():
    """测试0和负数的处理"""
    print("=== 0和负数处理测试 ===")
    test_cases = [-5, -1, 0, 1, 2]
    
    for num in test_cases:
        triton_result = triton.next_power_of_2(num) if num > 0 else "N/A"
        safe_result = safe_next_power_of_2(num)
        exp2_fixed = exp2_upper_fixed(num)
        print(f"{num:2d} -> triton: {str(triton_result):4s}, safe: {safe_result:4d}, exp2_fixed: {exp2_fixed:4d}")

def test_basic_cases():
    """测试基本用例"""
    print("=== 基本测试用例 ===")
    test_cases = [1023, 1024, 1025, 256, 257, 512, 513, 128, 129]
    
    for num in test_cases:
        triton_result = triton.next_power_of_2(num)
        exp2_result = exp2_upper(num)
        status = "✓" if triton_result == exp2_result else "✗"
        print(f"{num:4d} -> triton: {triton_result:4d}, exp2_upper: {exp2_result:4d} {status}")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 边界测试用例 ===")
    edge_cases = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256]
    
    for num in edge_cases:
        triton_result = triton.next_power_of_2(num)
        exp2_result = exp2_upper(num)
        status = "✓" if triton_result == exp2_result else "✗"
        print(f"{num:3d} -> triton: {triton_result:4d}, exp2_upper: {exp2_result:4d} {status}")

def test_large_numbers():
    """测试大数值"""
    print("\n=== 大数值测试 ===")
    large_cases = [1000000, 1048576, 1048577, 2097152, 16777216, 33554432]
    
    for num in large_cases:
        triton_result = triton.next_power_of_2(num)
        exp2_result = exp2_upper(num)
        status = "✓" if triton_result == exp2_result else "✗"
        print(f"{num:8d} -> triton: {triton_result:8d}, exp2_upper: {exp2_result:8d} {status}")

def test_power_of_2():
    """测试2的幂次"""
    print("\n=== 2的幂次测试 ===")
    powers = [2**i for i in range(1, 21)]
    
    for num in powers:
        triton_result = triton.next_power_of_2(num)
        exp2_result = exp2_upper(num)
        status = "✓" if triton_result == exp2_result else "✗"
        print(f"2^{num.bit_length()-1:2d} = {num:7d} -> triton: {triton_result:7d}, exp2_upper: {exp2_result:7d} {status}")

def test_expert_related_numbers():
    """测试MoE相关的专家数量"""
    print("\n=== MoE专家数量相关测试 ===")
    expert_cases = [64, 128, 256, 512, 1024, 127, 255, 511, 1023]
    
    for num in expert_cases:
        triton_result = triton.next_power_of_2(num)
        exp2_result = exp2_upper(num)
        status = "✓" if triton_result == exp2_result else "✗"
        print(f"experts={num:4d} -> triton: {triton_result:4d}, exp2_upper: {exp2_result:4d} {status}")

def performance_test():
    """性能测试"""
    print("\n=== 性能测试 ===")
    test_nums = list(range(1, 10000, 100))
    
    # 测试 triton.next_power_of_2
    start_time = time.time()
    for num in test_nums:
        triton.next_power_of_2(num)
    triton_time = time.time() - start_time
    
    # 测试 exp2_upper
    start_time = time.time()
    for num in test_nums:
        exp2_upper(num)
    exp2_time = time.time() - start_time
    
    print(f"triton.next_power_of_2: {triton_time:.6f}s")
    print(f"exp2_upper:             {exp2_time:.6f}s")
    print(f"性能比例: {exp2_time/triton_time:.2f}x (exp2_upper vs triton)")

def test_consistency():
    """一致性测试 - 随机数测试"""
    print("\n=== 一致性测试 ===")
    import random
    random.seed(42)
    
    test_count = 100
    mismatches = 0
    
    for i in range(test_count):
        num = 0
        triton_result = triton.next_power_of_2(num)
        exp2_result = exp2_upper(num)
        
        if triton_result != exp2_result:
            mismatches += 1
            print(f"不匹配: {num} -> triton: {triton_result}, exp2_upper: {exp2_result}")
    
    print(f"测试 {test_count} 个随机数，发现 {mismatches} 个不匹配")
    print(f"一致性: {(test_count-mismatches)/test_count*100:.1f}%")

if __name__ == "__main__":
    test_zero_and_negative()
    print()
    test_basic_cases()
    test_edge_cases()
    test_large_numbers()
    test_power_of_2()
    test_expert_related_numbers()
    performance_test()
    test_consistency()
