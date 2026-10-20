# -*- coding: utf-8 -*-
"""
虚拟聚合模型轮询机制与自适应熔断算法单元测试
"""
import time
import os
import sys
from pathlib import Path

# 加入 gateway 目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import virtual_models as vm

def test_round_robin_rotation():
    print(">>> 测试 1: 聚合模型 Round-Robin 平滑轮询调度")
    original = vm.get_virtual_models(use_cache=False)
    try:
        test_vmodel = {
            "id": "test-rr-model",
            "display_name": "Test RR Model",
            "chain": ["provider1::modelA", "provider2::modelB", "provider3::modelC"],
            "routing_strategy": "round-robin",
            "enabled": True
        }
        vm.save_virtual_models([test_vmodel])

        c1 = vm.get_ordered_chain("test-rr-model")
        c2 = vm.get_ordered_chain("test-rr-model")
        c3 = vm.get_ordered_chain("test-rr-model")
        c4 = vm.get_ordered_chain("test-rr-model")

        assert c1[0] == "provider1::modelA", f"Expected modelA first, got {c1[0]}"
        assert c2[0] == "provider2::modelB", f"Expected modelB first, got {c2[0]}"
        assert c3[0] == "provider3::modelC", f"Expected modelC first, got {c3[0]}"
        assert c4[0] == "provider1::modelA", f"Expected modelA first again, got {c4[0]}"
        print("  [PASS] 轮询机制依次平滑轮转各个候选模型！")
    finally:
        vm.save_virtual_models(original)


def test_priority_strategy():
    print(">>> 测试 2: 聚合模型 Priority 主备模式")
    original = vm.get_virtual_models(use_cache=False)
    try:
        test_vmodel = {
            "id": "test-pri-model",
            "display_name": "Test Priority Model",
            "chain": ["provider1::modelA", "provider2::modelB", "provider3::modelC"],
            "routing_strategy": "priority",
            "enabled": True
        }
        vm.save_virtual_models([test_vmodel])

        c1 = vm.get_ordered_chain("test-pri-model")
        c2 = vm.get_ordered_chain("test-pri-model")
        assert c1[0] == "provider1::modelA"
        assert c2[0] == "provider1::modelA"
        print("  [PASS] 主备优先模式始终首选最高优先级候选！")
    finally:
        vm.save_virtual_models(original)


def test_consecutive_failure_cooldown():
    print(">>> 测试 3: 连续失败即时熔断算法 (解决超长耗时跨越窗口导致的无法熔断缺陷)")
    m = "provider_bad::fail_model"
    vm.record_model_success(m)
    assert not vm.is_model_cooling(m)

    # 第 1 次普通失败
    tripped1 = vm.record_model_failure(m, reason="502_bad_gateway", is_severe=False)
    assert not tripped1, "第 1 次普通失败不应立即触发熔断"
    assert not vm.is_model_cooling(m)

    # 第 2 次普通失败（连续2次失败）
    tripped2 = vm.record_model_failure(m, reason="502_bad_gateway", is_severe=False)
    assert tripped2, "连续 2 次失败必须立即触发软熔断！"
    assert vm.is_model_cooling(m), "模型当前必须处于冷却中！"
    print(f"  [PASS] 连续2次失败成功触发熔断，冷却剩余时间: {vm.cooldown_remaining(m):.1f}s")

    # 成功恢复测试
    vm.record_model_success(m)
    assert not vm.is_model_cooling(m)
    print("  [PASS] 成功调用后立即重置连续失败并解除熔断！")


def test_severe_failure_immediate_cooldown():
    print(">>> 测试 4: 致命严重故障（如 SSLError、503无渠道、网络中断）单次即刻熔断")
    m = "provider_ssl::broken_model"
    vm.record_model_success(m)

    # 单次严重致命故障
    tripped = vm.record_model_failure(m, reason="SSLError: decryption failed", is_severe=True)
    assert tripped, "严重致命故障必须第 1 次就立即触发熔断！"
    assert vm.is_model_cooling(m), "严重故障后模型必须处于冷却状态！"
    print(f"  [PASS] 严重故障即刻触发熔断，冷却剩余: {vm.cooldown_remaining(m):.1f}s")
    vm.record_model_success(m)


if __name__ == "__main__":
    test_round_robin_rotation()
    test_priority_strategy()
    test_consecutive_failure_cooldown()
    test_severe_failure_immediate_cooldown()
    print("\n===============================")
    print("ALL ROUTING & SWITCHING TESTS PASSED!")
    print("===============================")
