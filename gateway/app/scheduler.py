"""
SurgeScheduler - 浪涌调度器核心（严格公平调度 + 桶级健康守护）

=== 调度机制 ===
- 滑动窗口计数：每个 (key_id, model) 桶维护请求/成败/延迟等计数，是所有调度决策的数据来源
- 429 桶级冷却：上游 429 → 桶级冷却（默认 240 秒，可在系统设置调整），期间不参与候选
- 模型级 429 熔断：30 秒窗口内同一模型 429 ≥ 20 次 → 直接向客户端返回 429，10 秒后解除
- 连接失败隔离：连续 3 次连接失败/SSL 错误 → 该桶隔离 5 分钟（权重归零）
- 5xx 退避：连续 5xx 逐步衰减健康评分，影响健康性过滤
- RPM 限制：桶 RPM 超限 → 标记软繁忙，跳过本轮候选
- 冷密钥预热：新启用/久未用的组合渐进恢复流量
- 健康度评分：后台周期综合成功率/延迟计算，作为健康性过滤输入

=== 桶级冷却强制约束（不可违反） ===
1. 冷却状态只存储在 (key_id, model) 复合桶级别，不存在任何密钥级冷却状态字段
2. select_key 检查冷却状态时只使用复合键查询，不存在任何密钥级查询路径
3. 软繁忙与冷却完全分离：RPM 超限只标记桶级软繁忙，永不调用冷却函数
4. 后台异步任务只操作桶级状态，不修改任何密钥级聚合状态

=== select_key 流程 ===
1. 获取所有活跃密钥
2. 硬性过滤：冷却、隔离、软繁忙、密钥解密失败
3. 健康性评分过滤：评分 < 0.1 视为不健康，排除
4. 严格公平选择：在健康候选中，按"近期使用次数最少 → 最近使用时间最久"排序
5. 最小候选池保障：若健康候选 < 3，放宽 predicted_busy/soft_busy 限制
6. 记录使用并返回
"""
import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import httpx

from app.database import fetch_all, fetch_one, get_setting, execute
from app.security import decrypt_upstream_key

logger = logging.getLogger("acu.scheduler")


# ========== 模型特性配置 ==========

# 慢速模型（首字节延迟高，需要更长超时）
# 仅包含实测可用的模型
SLOW_MODELS = {
    "z-ai/glm-5.2",
    "minimaxai/minimax-m3",
    "minimaxai/minimax-m2.7",
    "qwen/qwen3.5-397b-a17b",
    "nvidia/nemotron-3-ultra-550b-a55b",
}

# 慢速模型差异化切换阈值（统一38，确保所有模型容量一致）
# 仅包含实测可用的模型
SLOW_MODEL_THRESHOLDS = {
    "z-ai/glm-5.2": 38,
    "minimaxai/minimax-m3": 38,
    "minimaxai/minimax-m2.7": 38,
    "qwen/qwen3.5-397b-a17b": 38,
    "nvidia/nemotron-3-ultra-550b-a55b": 38,
}

# 推理模型（返回 reasoning_content 字段）
REASONING_MODELS = {
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
}

# 模型默认参数（基于NVIDIA NIM使用示例）
MODEL_DEFAULTS = {
    "z-ai/glm-5.2": {"temperature": 1, "top_p": 1, "max_tokens": 16384},
    "minimaxai/minimax-m3": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192},
    "minimaxai/minimax-m2.7": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192},
    "deepseek-ai/deepseek-v4-pro": {"temperature": 1, "top_p": 0.95, "max_tokens": 16384},
    "deepseek-ai/deepseek-v4-flash": {"temperature": 1, "top_p": 0.95, "max_tokens": 16384},
}

# 超时配置
DEFAULT_TIMEOUT = 180.0        # 默认3分钟（Agent工作流需要较长时间）
SLOW_MODEL_TIMEOUT = 300.0     # 慢速模型5分钟
STREAM_FIRST_BYTE_TIMEOUT = 120.0  # 流式首字节2分钟
STREAM_IDLE_TIMEOUT = 180.0    # 流式空闲3分钟


def get_threshold_for_model(model: str, default: int = 38) -> int:
    """返回模型的有效切换阈值"""
    return SLOW_MODEL_THRESHOLDS.get(model, default)


def get_timeout_for_model(model: str) -> float:
    """根据模型返回合适的整体超时时间"""
    return SLOW_MODEL_TIMEOUT if model in SLOW_MODELS else DEFAULT_TIMEOUT


def apply_model_defaults(body: dict) -> dict:
    """根据NVIDIA NIM使用示例自动补全模型默认参数（不覆盖用户设置）"""
    model = body.get("model", "")
    defaults = MODEL_DEFAULTS.get(model)
    if defaults:
        for key, value in defaults.items():
            if key not in body:
                body[key] = value
    return body


# ========== 分桶数据结构 ==========

@dataclass
class BucketData:
    """
    单个复合桶 (key_id, model) 的完整状态数据

    字段按机制划分（字段隔离原则）：
    - 滑动窗口计数: timestamps, response_times, total_*, last_*_at, last_failure_type
    - 自适应阈值: dynamic_threshold
    - 冷却: cooldown_until, cooldown_history
    - 5xx 退避: consecutive_5xx, first_5xx_time
    - 隔离: isolation_until, consecutive_conn_fail
    - 健康度: health_score, health_history
    - 预热: warmup_progress
    - 自愈状态: heal_action, heal_until
    """
    # === 分桶滑动窗口计数器 ===
    timestamps: deque = field(default_factory=lambda: deque(maxlen=120))
    response_times: deque = field(default_factory=lambda: deque(maxlen=50))
    total_requests: int = 0
    total_success: int = 0
    total_failures: int = 0
    total_429: int = 0
    total_5xx: int = 0
    total_timeout: int = 0
    total_conn_error: int = 0
    total_ssl_error: int = 0
    last_success_at: Optional[float] = None
    last_failure_at: Optional[float] = None
    last_failure_type: Optional[str] = None

    # === 自适应并发阈值 ===
    dynamic_threshold: int = 38

    # === 差异化冷却时长 ===
    cooldown_until: float = 0.0
    cooldown_history: deque = field(default_factory=lambda: deque(maxlen=20))
    consecutive_403: int = 0  # 连续403计数（用于自适应退避）

    # === 5xx 退避权重衰减 ===
    consecutive_5xx: int = 0
    first_5xx_time: Optional[float] = None

    # === 连接失败隔离 ===
    isolation_until: float = 0.0
    consecutive_conn_fail: int = 0

    # === 健康度评分 ===
    health_score: float = 100.0
    health_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # === 冷密钥渐进预热 ===
    warmup_progress: int = 30  # 默认30表示已预热（新桶不需要预热）

    # ===智能异常自愈引擎 ===
    heal_action: Optional[str] = None
    heal_until: float = 0.0

    # ===自适应负载预判器（趋势数据） ===
    rpm_trend: deque = field(default_factory=lambda: deque(maxlen=60))

    def get_rpm(self) -> int:
        """计算当前60秒窗口内的RPM"""
        now = time.time()
        while self.timestamps and now - self.timestamps[0] > 60:
            self.timestamps.popleft()
        return len(self.timestamps)

    def get_success_rate(self) -> float:
        """计算成功率（0~100）

        修复：使用total_requests作为分母，避免部分请求状态未知时成功率偏高
        """
        if self.total_requests == 0:
            return 100.0
        return (self.total_success / self.total_requests) * 100.0

    def get_p95_rt(self) -> float:
        """计算P95响应时间（秒）"""
        if len(self.response_times) < 5:
            return 0.0
        sorted_rt = sorted(self.response_times)
        idx = int(len(sorted_rt) * 0.95)
        return sorted_rt[min(idx, len(sorted_rt) - 1)]

    def get_avg_rt(self) -> float:
        """计算平均响应时间（秒）"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def is_cooled_down(self) -> bool:
        """检查桶是否在冷却中"""
        return time.time() < self.cooldown_until

    def is_isolated(self) -> bool:
        """检查桶是否被隔离"""
        return time.time() < self.isolation_until

    def is_healing(self) -> bool:
        """检查桶是否在自愈中"""
        return self.heal_action is not None and time.time() < self.heal_until


# ========== 客户端治理状态（只监测不拦截） ==========

# v9.2: 并发请求超时时间（秒）- 超过此时间的请求视为已释放
# 防止客户端断连但服务端未收到释放信号导致的并发计数不准确
CONCURRENCY_REQUEST_TIMEOUT = 120.0  # 120秒无响应视为已释放


@dataclass
class ClientMetrics:
    """客户端行为监测数据（v9.2 升级：毫秒级精准并发追踪）"""
    inflight_count: int = 0  # 当前并发数
    # v9.2: 毫秒级精准追踪 - 记录每个并发请求的开始时间戳
    # 用于超时检测和高精度并发计算
    inflight_timestamps: list = field(default_factory=list)  # [start_time, ...]
    burst_timestamps: deque = field(default_factory=lambda: deque(maxlen=200))  # 突发检测（扩大容量）
    daily_count: int = 0  # 日用量
    daily_reset_at: float = 0.0
    request_intervals: deque = field(default_factory=lambda: deque(maxlen=50))  # 请求间隔（商用识别用）
    last_request_at: float = 0.0
    model_switches: deque = field(default_factory=lambda: deque(maxlen=20))  # 模型切换记录
    last_model: str = ""

    def get_burst_count(self, window_seconds: float = 3.0) -> int:
        """计算指定时间窗口内的突发请求数（修复：原实现未过滤时间窗口）"""
        now = time.time()
        while self.burst_timestamps and now - self.burst_timestamps[0] > window_seconds:
            self.burst_timestamps.popleft()
        return len(self.burst_timestamps)

    def record_inflight_start(self) -> float:
        """记录并发请求开始，返回开始时间戳（毫秒精度）"""
        now = time.time()
        self.inflight_count += 1
        self.inflight_timestamps.append(now)
        return now

    def record_inflight_end(self) -> float:
        """
        记录并发请求结束。
        自动清理超时的陈旧请求记录。
        返回本次请求的耗时（秒）。
        """
        now = time.time()
        self.inflight_count = max(0, self.inflight_count - 1)
        # 清理最早的一个时间戳
        if self.inflight_timestamps:
            start = self.inflight_timestamps.pop(0)
            return now - start
        return 0.0

    def cleanup_stale_inflight(self) -> int:
        """
        清理超时的陈旧并发请求（毫秒级精准检测）。
        返回清理的陈旧请求数量。
        """
        now = time.time()
        stale_count = 0
        # 从前往后清理超时的陈旧的请求
        while self.inflight_timestamps and (now - self.inflight_timestamps[0]) > CONCURRENCY_REQUEST_TIMEOUT:
            self.inflight_timestamps.pop(0)
            self.inflight_count = max(0, self.inflight_count - 1)
            stale_count += 1
        return stale_count

    def get_real_inflight_count(self) -> int:
        """
        获取真实的并发数（经超时清理后）。
        确保并发计数准确反映实际在途请求。
        """
        self.cleanup_stale_inflight()
        return self.inflight_count


# ========== 全局调度器状态 ==========

@dataclass
class SchedulerStats:
    """调度器全局统计（用于监控）"""
    total_select_calls: int = 0
    total_none_returns: int = 0  # 返回None的次数（所有密钥耗尽）
    total_cooldowns_triggered: int = 0
    total_isolations_triggered: int = 0
    total_heal_actions: int = 0
    degraded_mode: bool = False  # 全局降级模式


# ========== SurgeScheduler 调度器核心 ==========

class SurgeScheduler:
    """
    浪涌调度器 v10.0

    核心架构：
    1. 所有状态以 (key_id, model) 复合桶为键存储
    2. 软繁忙（运行时集合）与冷却（cooldown_until字段）完全分离
    3. select_key 只使用复合键查询冷却状态
    4. 后台任务只操作桶级字段
    """

    POOL_MAX_CONNECTIONS = 300
    POOL_MAX_KEEPALIVE = 150

    # 活跃密钥缓存TTL（秒）
    ACTIVE_KEYS_CACHE_TTL = 30.0

    # 冷却时长（秒）- 差异化冷却策略
    # 不同失败类型的冷却时长不同：
    # - 429：5秒（NVIDIA限流通常1-2秒，5秒已足够恢复）
    # - 403：60秒（权限拒绝，短时间内不会恢复，需长时间避免重复选择）
    # - timeout：15秒（可能是暂时性延迟，中等冷却即可）
    COOLDOWN_429_SECONDS = 5       # 429冷却：快速恢复
    COOLDOWN_403_SECONDS = 60      # 403冷却：长时间避免（权限拒绝型密钥）
    COOLDOWN_TIMEOUT_SECONDS = 15  # 超时冷却：中等恢复
    COOLDOWN_SECONDS = 5           # 默认冷却（向后兼容）
    # 403自适应退避：同一(key,model)连续403时指数增长冷却时长
    COOLDOWN_403_MAX_SECONDS = 600  # 403最大冷却10分钟（彻底避免100%失败密钥被反复选择）
    # 隔离时长（秒）
    ISOLATION_SECONDS = 300  # 5分钟（超时可能是网络抖动，不需要30分钟那么长）
    # 5xx恢复时长（秒）
    XX5_RECOVERY_SECONDS = 600  # 10分钟
    # 预热请求数
    WARMUP_TARGET = 30

    # 模型级429熔断：当某模型在短时间内收到大量429时，暂停该模型的调度
    MODEL_429_CIRCUIT_BREAKER_THRESHOLD = 20  # 30秒内429次数阈值
    MODEL_429_CIRCUIT_BREAKER_WINDOW = 30.0   # 统计窗口（秒）
    MODEL_429_CIRCUIT_BREAKER_COOLDOWN = 10.0  # 熔断冷却时间（秒）

    # 模型级5xx熔断：当某模型在短时间内收到大量5xx时，暂停该模型的调度
    # 提高容错性 - 阈值从10提高到30，窗口缩短到20s，cooldown从30s降到10s
    # 确保5xx不会导致模型被长时间禁止调用
    MODEL_5XX_CIRCUIT_BREAKER_THRESHOLD = 30   # 20秒内5xx次数阈值（大幅放宽）
    MODEL_5XX_CIRCUIT_BREAKER_WINDOW = 20.0    # 统计窗口（秒）
    MODEL_5XX_CIRCUIT_BREAKER_COOLDOWN = 10.0  # 熔断冷却时间（秒）- 快速恢复

    # 趋势感知自适应均衡
    TRAE_TREND_WINDOW = 30.0  # 趋势计算窗口（秒）

    # 数据库日志自动清理计数器（每6小时执行一次）
    _cleanup_counter: int = 0
    CLEANUP_INTERVAL_CYCLES = 2160  # 10秒×2160=6小时
    TRAE_HIGH_TREND_MULTIPLIER = 0.6  # 高上升趋势乘数
    TRAE_LOW_TREND_MULTIPLIER = 1.3  # 下降趋势乘数

    # 周期性弹性调度
    LOBSTER_MOLT_SCORE_MULTIPLIER = 0.5  # 脱壳期评分乘数

    def __init__(self):
        # === 复合桶状态（键为 (key_id, model) 元组） ===
        self._buckets: Dict[Tuple[str, str], BucketData] = defaultdict(BucketData)

        # === 软繁忙标记（运行时集合，与冷却完全分离） ===
        self._soft_busy: set = set()  # 元素为 (key_id, model) 元组

        # === 负载预判排除集合（运行时集合） ===
        self._predicted_busy: set = set()

        # === 密钥缓存（key_id -> api_key明文） ===
        # 安全修复: 添加TTL过期机制，避免明文密钥永久驻留内存
        self._key_cache: Dict[str, str] = {}
        self._key_cache_timestamps: Dict[str, float] = {}  # key_id -> 缓存写入时间戳
        self.KEY_CACHE_TTL = 300.0  # 5分钟过期，平衡安全性与性能

        # === 活跃密钥内存缓存（5秒TTL） ===
        self._active_keys_cache = []
        self._active_keys_cache_time = 0.0

        # === 客户端治理数据 ===
        self._client_metrics: Dict[str, ClientMetrics] = defaultdict(ClientMetrics)

        # === 全局统计 ===
        self._stats = SchedulerStats()

        # === 模型级429熔断状态 ===
        # model -> deque of timestamps (429发生时间)
        self._model_429_timestamps: dict = defaultdict(lambda: deque(maxlen=200))
        # model -> 熔断截止时间
        self._model_429_circuit_breaker: dict = {}

        # === 模型级5xx熔断状态 ===
        # model -> deque of timestamps (5xx发生时间)
        self._model_5xx_timestamps: dict = defaultdict(lambda: deque(maxlen=100))
        # model -> 熔断截止时间
        self._model_5xx_circuit_breaker: dict = {}

        # === HTTP连接池 ===
        self._http_pool: Optional[httpx.AsyncClient] = None
        self._stream_pool: Optional[httpx.AsyncClient] = None

        # === 后台任务最后执行时间 ===
        self._last_threshold_update = 0.0
        self._last_health_update = 0.0
        self._last_heal_check = 0.0
        self._threshold_update_count = 0
        self._health_update_count = 0
        self._heal_check_count = 0

        # === 健康密钥计数 ===
        self._healthy_key_count = 0

        # === 客户端密钥缓存（30秒TTL，减少 authenticate_client 的DB查询） ===
        self._client_key_cache: Dict[str, dict] = {}
        self._client_key_cache_time: float = 0.0
        self.CLIENT_KEY_CACHE_TTL = 30.0

        # === 上游键名称缓存（减少 get_algorithm_detail 的DB查询） ===
        self._upstream_key_names: Dict[str, str] = {}
        self._upstream_key_cache_time: float = 0.0
        self.UPSTREAM_KEY_CACHE_TTL = 30.0

        # === 客户端名称缓存（减少 get_algorithm_detail 的DB查询） ===
        self._client_name_cache: Dict[str, str] = {}
        self._client_name_cache_time: float = 0.0
        self.CLIENT_NAME_CACHE_TTL = 30.0

        # ===龙虾脱壳式弹性调度状态 ===
        self._molting_keys: Dict[str, float] = {}  # key_id -> molt_until timestamp
        self._last_lobster_check: float = 0.0

        # === 严格公平调度（Strict Fair Dispatch） ===
        # 滑动窗口近期计数（300秒窗口，替代累计计数避免历史偏差）
        self._fair_window_seconds = 300  # 5分钟公平窗口
        self._bucket_recent_usage: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=200))
        self._key_recent_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._bucket_last_used: Dict[Tuple[str, str], float] = {}  # (key_id, model) -> last_used_timestamp
        self._key_last_used: Dict[str, float] = {}  # key_id -> last_used_timestamp
        # 全局轮询序号（严格递增，确保同窗口内按到达顺序轮询）
        self._dispatch_sequence: int = 0
        self._bucket_dispatch_seq: Dict[Tuple[str, str], int] = {}  # 最近一次分配序号

        # === 失效密钥自动检测器 ===
        # v10.0: 连续5次密钥认证失败自动停用，避免死密钥消耗调度资源
        self._key_consecutive_auth_fail: Dict[str, int] = {}  # key_id -> 连续403/401计数
        self._auto_deactivated_keys: Dict[str, str] = {}  # key_id -> reason (已自动停用的密钥)
        self._AUTH_FAIL_DEACTIVATE_THRESHOLD = 5  # 连续5次认证失败自动停用

        # === 系统资源监控（内存感知调度） ===
        self._memory_pressure: float = 0.0  # 0.0~1.0 内存压力系数
        self._last_resource_check: float = 0.0
        self._memory_saving_mode: bool = False

        logger.info("SurgeScheduler 初始化完成（严格公平调度 + 桶级健康守护）")

    # ========== 系统资源监控（内存感知调度） ==========

    async def _check_system_resources(self):
        """
        读取 /proc/meminfo 获取可用内存，计算内存压力系数

        每30秒检查一次，结果存储在 self._memory_pressure 中
        """
        now = time.time()
        if now - self._last_resource_check < 30.0:
            return
        self._last_resource_check = now
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()

            mem_available = 0
            mem_total = 0
            for line in meminfo.splitlines():
                if line.startswith("MemAvailable:"):
                    # 单位: kB
                    mem_available = int(line.split()[1]) / 1024  # 转换为MB
                elif line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) / 1024  # 转换为MB

            if mem_total == 0:
                self._memory_pressure = 0.0
                return

            # 计算压力系数
            if mem_available < 100:
                self._memory_pressure = 1.0  # 临界
            elif mem_available < 200:
                # 可用内存100~200MB，压力0.8~0.95
                self._memory_pressure = 0.8 + (1.0 - (mem_available - 100) / 100) * 0.15
            else:
                # 基于可用内存比例计算（可用/总量），反向映射到0~0.8
                ratio = mem_available / mem_total
                self._memory_pressure = max(0.0, min(0.8, 1.0 - ratio * 2.0))

            # 更新内存节约模式
            old_mode = self._memory_saving_mode
            self._memory_saving_mode = self._memory_pressure > 0.7
            if old_mode != self._memory_saving_mode:
                logger.info(
                    f"系统内存{'进入' if self._memory_saving_mode else '退出'}节约模式: "
                    f"pressure={self._memory_pressure:.2f}, "
                    f"available={mem_available:.0f}MB/{mem_total:.0f}MB"
                )

        except Exception as e:
            logger.debug(f"读取系统内存信息失败: {e}")
            self._memory_pressure = 0.0


    # ========== 桶操作基础方法 ==========

    def _get_bucket(self, key_id: str, model: str) -> BucketData:
        """获取桶数据（不存在则自动创建）"""
        return self._buckets[(key_id, model)]

    def _get_active_keys(self) -> List[dict]:
        """从数据库获取所有活跃上游密钥（5秒内存缓存）"""
        now = time.time()
        if self._active_keys_cache and now - self._active_keys_cache_time < self.ACTIVE_KEYS_CACHE_TTL:
            return self._active_keys_cache
        self._active_keys_cache = fetch_all(
            "SELECT id, name, api_key_ciphertext, key_prefix, weight, "
            "rpm_limit, switch_threshold, status, provider, base_url "
            "FROM upstream_keys WHERE status = 'active'"
        )
        self._active_keys_cache_time = now
        return self._active_keys_cache

    def invalidate_active_keys_cache(self):
        """使活跃密钥缓存失效（密钥增删改时调用）"""
        self._active_keys_cache = []
        self._active_keys_cache_time = 0.0


    def get_cached_key(self, key_id: str) -> Optional[str]:
        """获取缓存的密钥明文（检查TTL过期）"""
        if key_id not in self._key_cache:
            return None
        # 检查是否过期
        cached_at = self._key_cache_timestamps.get(key_id, 0)
        if time.time() - cached_at > self.KEY_CACHE_TTL:
            # 过期，清除缓存
            self._key_cache.pop(key_id, None)
            self._key_cache_timestamps.pop(key_id, None)
            return None
        return self._key_cache.get(key_id)

    def invalidate_key_cache(self, key_id: str):
        """使密钥缓存失效"""
        self._key_cache.pop(key_id, None)
        self._key_cache_timestamps.pop(key_id, None)


    def _ensure_key_cached(self, key_id: str, ciphertext: str) -> str:
        """确保密钥已缓存，返回明文"""
        # 检查缓存是否存在且未过期
        cached = self.get_cached_key(key_id)
        if cached is not None:
            return cached
        if not ciphertext:
            logger.warning(f"密钥解密失败: key={key_id[:8]} ciphertext为空")
            return ""
        master_key = get_setting("upstream_master_key")
        if master_key:
            try:
                plaintext = decrypt_upstream_key(ciphertext, master_key)
                self._key_cache[key_id] = plaintext
                self._key_cache_timestamps[key_id] = time.time()
            except Exception as e:
                logger.error(f"密钥解密异常: key={key_id[:8]} error={e}")
                return ""
        return self._key_cache.get(key_id, "")

    # ========== HTTP连接池 ==========

    async def _ensure_pools(self):
        """确保连接池已创建"""
        if self._http_pool is None or self._http_pool.is_closed:
            limits = httpx.Limits(
                max_connections=self.POOL_MAX_CONNECTIONS,
                max_keepalive_connections=self.POOL_MAX_KEEPALIVE,
                keepalive_expiry=120,
            )
            self._http_pool = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_TIMEOUT),
                limits=limits,
                http2=False,
            )
        if self._stream_pool is None or self._stream_pool.is_closed:
            stream_limits = httpx.Limits(
                max_connections=self.POOL_MAX_CONNECTIONS,
                max_keepalive_connections=self.POOL_MAX_KEEPALIVE,
                keepalive_expiry=120,
            )
            # 连接池超时设置为600秒（大于per-chunk空闲超时180秒）
            # 让per-chunk空闲检测先于httpx超时触发，确保优雅终止
            self._stream_pool = httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=30.0),
                limits=stream_limits,
                http2=False,
            )

    async def get_http_pool(self) -> httpx.AsyncClient:
        await self._ensure_pools()
        return self._http_pool

    async def get_stream_pool(self) -> httpx.AsyncClient:
        await self._ensure_pools()
        return self._stream_pool

    def get_resource_status(self) -> dict:
        """获取中央智能资源状态仪表盘数据"""
        now = time.time()
        return {
            "memory_pressure": round(self._memory_pressure, 3),
            "memory_saving_mode": self._memory_saving_mode,
            "active_keys_cache_age": round(now - self._active_keys_cache_time, 1) if self._active_keys_cache else -1,
            "total_buckets": len(self._buckets),
            "total_soft_busy": len(self._soft_busy),
            "degraded_mode": self._stats.degraded_mode,
            "http_pool_active": self._http_pool is not None and not self._http_pool.is_closed,
            "stream_pool_active": self._stream_pool is not None and not self._stream_pool.is_closed,
            "inflight_requests": sum(m.inflight_count for m in self._client_metrics.values()),
            "cooling_buckets": sum(1 for b in self._buckets.values() if b.is_cooled_down()),
            "isolated_buckets": sum(1 for b in self._buckets.values() if b.is_isolated()),
        }

    # ========== 分桶滑动窗口计数器 ==========

    def record_response(self, key_id: str, model: str, success: bool,
                        rt_seconds: float, status_code: int, failure_type: str = ""):
        """
        记录响应数据 - 所有调度决策的唯一数据来源

        参数：
        - key_id: 上游密钥ID
        - model: 模型名称
        - success: 是否成功
        - rt_seconds: 响应时间（秒）
        - status_code: HTTP状态码
        - failure_type: 失败类型（""/"429"/"5xx"/"timeout"/"conn_error"/"ssl_error"）
        """
        bucket = self._get_bucket(key_id, model)
        now = time.time()

        # 记录时间戳（60秒滑动窗口）
        bucket.timestamps.append(now)

        # 记录响应时间
        if rt_seconds > 0:
            bucket.response_times.append(rt_seconds)

        # 更新总计数
        bucket.total_requests += 1

        # 更新成功/失败计数
        if success:
            bucket.total_success += 1
            # 成功时重置连续5xx计数
            bucket.consecutive_5xx = 0
            bucket.first_5xx_time = None
            # 成功时重置连接失败计数
            bucket.consecutive_conn_fail = 0
            # 成功时重置连续403计数
            bucket.consecutive_403 = 0
            bucket.last_success_at = now
        else:
            bucket.total_failures += 1
            bucket.last_failure_at = now
            bucket.last_failure_type = failure_type

            # 按状态码分类统计
            if status_code == 429 or failure_type == "429":
                bucket.total_429 += 1
            elif status_code >= 500 or failure_type == "5xx":
                bucket.total_5xx += 1
                # 更新连续5xx计数
                bucket.consecutive_5xx += 1
                # 每次新5xx都更新时间，确保10分钟恢复窗口基于最后一次错误而非第一次
                bucket.first_5xx_time = now
            elif failure_type == "timeout":
                bucket.total_timeout += 1
            elif failure_type == "conn_error":
                bucket.total_conn_error += 1
                # 更新连续连接失败计数
                bucket.consecutive_conn_fail += 1
            elif failure_type == "ssl_error":
                bucket.total_ssl_error += 1
                bucket.consecutive_conn_fail += 1

        # 记录RPM趋势
        bucket.rpm_trend.append(now)

        # 预热进度递增（成功时）
        if success and bucket.warmup_progress < self.WARMUP_TARGET:
            bucket.warmup_progress += 1

        # === 失效密钥自动检测 ===
        # v10.0: 连续5次认证失败自动停用，从调度池移除
        if not success and (status_code == 403 or status_code == 401):
            self._key_consecutive_auth_fail[key_id] = self._key_consecutive_auth_fail.get(key_id, 0) + 1
            count = self._key_consecutive_auth_fail[key_id]
            if count >= self._AUTH_FAIL_DEACTIVATE_THRESHOLD > 0 and key_id not in self._auto_deactivated_keys:
                self._auto_deactivated_keys[key_id] = f"连续{count}次认证失败自动停用"
                # 同步更新数据库
                try:
                    from app.database import execute as db_exec, utcnow
                    db_exec("UPDATE upstream_keys SET status = 'inactive' WHERE id = %s AND status = 'active'", (key_id,))
                    logger.warning(f"密钥自动停用: key={key_id[:8]} 连续{count}次认证失败，已标记为inactive")
                except Exception as e:
                    logger.error(f"密钥自动停用数据库更新失败: {e}")
            else:
                logger.debug(f"密钥认证失败记录: key={key_id[:8]} consecutive={count}/{self._AUTH_FAIL_DEACTIVATE_THRESHOLD}")
        elif success:
            # 成功请求重置认证失败计数
            self._key_consecutive_auth_fail.pop(key_id, None)

        logger.debug(
            f"记录响应: key={key_id[:8]} model={model} success={success} "
            f"rt={rt_seconds:.2f}s status={status_code} type={failure_type}"
        )

    # ========== 软繁忙标记 ==========


    # ========== 自适应并发阈值 ==========


    # ========== 差异化冷却时长 ==========

    def trigger_hard_cooldown(self, key_id: str, model: str, reason: str = "429") -> int:
        """
        触发硬冷却 - 仅冷却触发429的那个 (key_id, model) 桶

        【强制约束】不得存在任何以单一密钥为键的冷却操作

        冷却策略：
        - 429：5秒快速恢复（NVIDIA限流通常1-2秒）
        - 403：60秒起步 + 自适应退避（连续403指数增长至600秒）
        - timeout：15秒中等恢复（可能是暂时性延迟）
        - 其他：5秒默认冷却

        403自适应退避算法：
        - 第1次403：60秒冷却
        - 第2次403：120秒冷却（60×2）
        - 第3次403：240秒冷却（60×4）
        - 第4次+：600秒冷却（上限，彻底避免100%失败密钥被反复选择）
        - 成功请求重置consecutive_403为0
        """
        if not model:
            logger.warning(f"trigger_hard_cooldown被调用但model为空，跳过(key_id={key_id})")
            return 0

        bucket = self._get_bucket(key_id, model)
        now = time.time()

        # 差异化冷却时长
        if reason == "429":
            cooldown_seconds = self.COOLDOWN_429_SECONDS
        elif reason == "403":
            # 403自适应退避：连续403时指数增长
            bucket.consecutive_403 += 1
            backoff_multiplier = min(2 ** (bucket.consecutive_403 - 1), 10)  # 1,2,4,8,10...
            cooldown_seconds = min(self.COOLDOWN_403_SECONDS * backoff_multiplier, self.COOLDOWN_403_MAX_SECONDS)
            if bucket.consecutive_403 >= 3:
                logger.warning(
                    f"403自适应退避: key={key_id[:8]} model={model} "
                    f"连续403={bucket.consecutive_403}次 冷却={cooldown_seconds}s"
                )
        elif reason == "timeout":
            cooldown_seconds = self.COOLDOWN_TIMEOUT_SECONDS
        else:
            cooldown_seconds = self.COOLDOWN_SECONDS

        # 记录冷却历史
        bucket.cooldown_history.append(now)

        # 设置冷却
        bucket.cooldown_until = now + cooldown_seconds
        self._stats.total_cooldowns_triggered += 1

        # 修复：重置预热进度，冷却恢复后进入渐进式预热流程
        bucket.warmup_progress = 0

        # 模型级429熔断检测：记录429时间戳
        if reason == "429":
            self._model_429_timestamps[model].append(now)
            # 检查是否触发熔断
            recent_429s = [t for t in self._model_429_timestamps[model] if now - t < self.MODEL_429_CIRCUIT_BREAKER_WINDOW]
            if len(recent_429s) >= self.MODEL_429_CIRCUIT_BREAKER_THRESHOLD:
                self._model_429_circuit_breaker[model] = now + self.MODEL_429_CIRCUIT_BREAKER_COOLDOWN
                logger.warning(
                    f"模型级429熔断触发: model={model} "
                    f"窗口内429={len(recent_429s)}次/{self.MODEL_429_CIRCUIT_BREAKER_WINDOW:.0f}s "
                    f"熔断{self.MODEL_429_CIRCUIT_BREAKER_COOLDOWN:.0f}s"
                )

        # 模型级5xx熔断检测：记录5xx时间戳
        if reason == "5xx":
            self._model_5xx_timestamps[model].append(now)
            recent_5xx = [t for t in self._model_5xx_timestamps[model] if now - t < self.MODEL_5XX_CIRCUIT_BREAKER_WINDOW]
            if len(recent_5xx) >= self.MODEL_5XX_CIRCUIT_BREAKER_THRESHOLD:
                self._model_5xx_circuit_breaker[model] = now + self.MODEL_5XX_CIRCUIT_BREAKER_COOLDOWN
                logger.warning(
                    f"模型级5xx熔断触发: model={model} "
                    f"窗口内5xx={len(recent_5xx)}次/{self.MODEL_5XX_CIRCUIT_BREAKER_WINDOW:.0f}s "
                    f"熔断{self.MODEL_5XX_CIRCUIT_BREAKER_COOLDOWN:.0f}s"
                )

        logger.info(
            f"触发桶级冷却: key={key_id[:8]} model={model} "
            f"时长={cooldown_seconds}s 原因={reason} (仅该模型桶受影响，已重置预热)"
        )
        return cooldown_seconds

    def unfreeze_bucket(self, key_id: str, model: str) -> bool:
        """手动解冻特定桶（管理后台用）"""
        bucket = self._get_bucket(key_id, model)
        bucket.cooldown_until = 0
        bucket.isolation_until = 0
        bucket.consecutive_5xx = 0
        bucket.first_5xx_time = None
        bucket.consecutive_conn_fail = 0
        bucket.heal_action = None
        bucket.heal_until = 0
        # 重置预热进度
        bucket.warmup_progress = self.WARMUP_TARGET
        logger.info(f"手动解冻桶: key={key_id[:8]} model={model}")
        return True

    def unfreeze_key_all_buckets(self, key_id: str) -> int:
        """解冻某密钥的所有桶（管理后台用）"""
        count = 0
        for (kid, model) in list(self._buckets.keys()):
            if kid == key_id:
                self.unfreeze_bucket(key_id, model)
                count += 1
        logger.info(f"解冻密钥所有桶: key={key_id[:8]} 共{count}个桶")
        return count

    # ========== 5xx 退避权重衰减 ==========

    def get_5xx_multiplier(self, bucket: BucketData) -> float:
        """
        计算5xx退避权重乘数

        consecutive_5xx=0 → 1.0
        consecutive_5xx=1 → 0.8
        consecutive_5xx=2 → 0.5
        consecutive_5xx=3 → 0.2
        consecutive_5xx≥4 → 0.0 (持续10分钟后自动恢复0.8)
        """
        if bucket.consecutive_5xx == 0:
            return 1.0
        if bucket.consecutive_5xx >= 4:
            # 检查是否已过10分钟恢复期
            if bucket.first_5xx_time and time.time() - bucket.first_5xx_time > self.XX5_RECOVERY_SECONDS:
                bucket.consecutive_5xx = 0
                bucket.first_5xx_time = None
                return 0.8
            return 0.0
        multipliers = {1: 0.8, 2: 0.5, 3: 0.2}
        return multipliers.get(bucket.consecutive_5xx, 0.0)

    # ========== 连接失败隔离 ==========

    def trigger_isolation(self, key_id: str, model: str, reason: str = "conn_error"):
        """触发区域故障隔离（30分钟权重归零）"""
        bucket = self._get_bucket(key_id, model)
        if bucket.consecutive_conn_fail >= 3:
            bucket.isolation_until = time.time() + self.ISOLATION_SECONDS
            self._stats.total_isolations_triggered += 1
            logger.warning(
                f"触发隔离: key={key_id[:8]} model={model} "
                f"时长={self.ISOLATION_SECONDS}s 原因={reason}"
            )

    def get_isolation_multiplier(self, bucket: BucketData) -> float:
        """计算隔离权重乘数"""
        return 0.0 if bucket.is_isolated() else 1.0

    # ========== 健康度评分 ==========


    # ========== 池化动态权重 ==========

    def get_base_weight_multiplier(self, bucket: BucketData) -> float:
        """
        计算基础权重乘数（基于健康度）

        base_weight = 0.5 + (health_score / 100) × 1.5
        范围: 0.5 (health=0) ~ 2.0 (health=100)
        """
        return 0.5 + (bucket.health_score / 100) * 1.5

    # ========== 负载趋势预判 ==========


    # ========== 冷密钥渐进预热 ==========

    def get_warmup_multiplier(self, bucket: BucketData) -> float:
        """
        计算预热权重乘数

        warmup_progress < 10 → 0.3
        warmup_progress 10~19 → 0.6
        warmup_progress 20~29 → 0.9
        warmup_progress ≥ 30 → 1.0
        """
        if bucket.warmup_progress < 10:
            return 0.3
        elif bucket.warmup_progress < 20:
            return 0.6
        elif bucket.warmup_progress < 30:
            return 0.9
        return 1.0


    # ========== 趋势感知均衡 ==========

    def get_trae_multiplier(self, key_id: str, model: str) -> float:
        """
        计算趋势感知乘数（Trae优化算法）

        通过对比最近30秒与之前30秒的RPM，计算趋势斜率：
        - 斜率 > 1.5（RPM快速上升）→ 降低乘数，避免过载（最低0.6）
        - 斜率 < 1.0（RPM下降）→ 提高乘数，吸引流量（最高1.3）
        - 斜率在1.0~1.5之间 → 线性插值
        """
        bucket = self._get_bucket(key_id, model)
        now = time.time()

        # 清理5分钟前的趋势数据
        recent = [t for t in bucket.rpm_trend if now - t < 300]

        if len(recent) < 5:
            return 1.0

        # 最近30秒的RPM
        recent_window = [t for t in recent if now - t < self.TRAE_TREND_WINDOW]
        recent_rpm = len(recent_window)

        # 之前30秒的RPM（30~60秒前）
        previous_window = [t for t in recent if self.TRAE_TREND_WINDOW <= now - t < self.TRAE_TREND_WINDOW * 2]
        previous_rpm = len(previous_window)

        if previous_rpm == 0:
            # 无历史数据时，如果当前有流量则轻微降权，否则返回1.0
            return 0.9 if recent_rpm > 0 else 1.0

        # 计算趋势斜率
        slope = recent_rpm / previous_rpm

        if slope > 1.5:
            # 快速上升趋势，降权防止过载
            return self.TRAE_HIGH_TREND_MULTIPLIER
        elif slope < 1.0:
            # 下降趋势，提权吸引流量
            # 线性插值：slope=0.0→1.3, slope=1.0→1.0
            multiplier = 1.0 + (1.0 - slope) * (self.TRAE_LOW_TREND_MULTIPLIER - 1.0)
            return min(self.TRAE_LOW_TREND_MULTIPLIER, multiplier)
        else:
            # slope在1.0~1.5之间，线性插值
            # slope=1.0→1.0, slope=1.5→0.6
            multiplier = 1.0 - (slope - 1.0) / 0.5 * (1.0 - self.TRAE_HIGH_TREND_MULTIPLIER)
            return max(self.TRAE_HIGH_TREND_MULTIPLIER, multiplier)

    # ========== 周期性弹性调度 ==========


    def get_lobster_multiplier(self, key_id: str) -> float:
        """
        计算龙虾脱壳乘数

        如果密钥正在脱壳期，返回0.5；否则返回1.0
        """
        molt_until = self._molting_keys.get(key_id)
        if molt_until is not None:
            if time.time() < molt_until:
                return self.LOBSTER_MOLT_SCORE_MULTIPLIER
            else:
                # 脱壳期已过，清理状态
                del self._molting_keys[key_id]
                logger.info(f"弹性调度状态恢复: key={key_id[:8]}")
        return 1.0

    # ========== 严格公平调度（Strict Fair Dispatch） ==========

    def get_fair_dispatch_count(self, key_id: str, model: str) -> Tuple[int, int]:
        """
        获取桶级和密钥级的近期使用次数（滑动窗口）

        返回: (bucket_recent_count, key_recent_count)
        - bucket_recent_count: 该(key_id, model)桶在最近300秒内的使用次数
        - key_recent_count: 该key_id在最近300秒内的总使用次数

        使用滑动窗口替代累计计数，避免历史偏差
        """
        now = time.time()
        cutoff = now - self._fair_window_seconds

        # 桶级近期计数
        bucket_key = (key_id, model)
        bucket_times = self._bucket_recent_usage.get(bucket_key, deque(maxlen=500))
        # 清理过期时间戳
        while bucket_times and bucket_times[0] < cutoff:
            bucket_times.popleft()
        bucket_count = len(bucket_times)

        # 密钥级近期计数
        key_times = self._key_recent_usage.get(key_id, deque(maxlen=2000))
        while key_times and key_times[0] < cutoff:
            key_times.popleft()
        key_count = len(key_times)

        return (bucket_count, key_count)

    def get_round_robin_multiplier(self, key_id: str, model: str) -> float:
        """
        兼容接口：返回1.0

        当前方案通过严格公平排序实现均衡。
        保留此方法以兼容calculate_score调用，始终返回1.0（不影响评分）。
        实际均衡由select_key中的公平排序逻辑实现。
        """
        return 1.0

    def record_key_usage(self, key_id: str, model: str = ""):
        """记录密钥被使用（在select_key选中后调用）

        使用滑动窗口时间戳队列替代累计计数器
        """
        now = time.time()
        self._dispatch_sequence += 1

        # 密钥级记录
        self._key_last_used[key_id] = now
        self._key_recent_usage[key_id].append(now)

        # 桶级记录
        if model:
            bucket_key = (key_id, model)
            self._bucket_last_used[bucket_key] = now
            self._bucket_recent_usage[bucket_key].append(now)
            self._bucket_dispatch_seq[bucket_key] = self._dispatch_sequence

    


    # ========== 客户端治理（只监测不拦截） ==========

    def record_client_request(self, client_id: str, model: str):
        """记录客户端请求（数据采集 + 毫秒级并发追踪）"""
        metrics = self._client_metrics[client_id]
        now = time.time()

        # v9.2: 毫秒级精准并发追踪
        metrics.record_inflight_start()
        # 同时清理超时的陈旧请求
        metrics.cleanup_stale_inflight()

        # 突发检测（添加时间戳，get_burst_count 会自动清理过期数据）
        metrics.burst_timestamps.append(now)

        # 日用量统计（使用当天凌晨作为重置时间）
        from datetime import datetime
        today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        # 如果从未重置过，或已经过了今天凌晨，则重置
        if metrics.daily_reset_at <= 0 or now >= metrics.daily_reset_at or metrics.daily_reset_at < today_midnight:
            metrics.daily_count = 0
            # 下次重置时间为明天凌晨
            metrics.daily_reset_at = today_midnight + 86400
        metrics.daily_count += 1

        # 请求间隔记录（商用识别用）
        if metrics.last_request_at > 0:
            interval = now - metrics.last_request_at
            metrics.request_intervals.append(interval)
        metrics.last_request_at = now

        # 模型切换记录
        if metrics.last_model and metrics.last_model != model:
            metrics.model_switches.append((now, model))
        metrics.last_model = model

    def release_client_request(self, client_id: str):
        """释放客户端并发计数（v9.2 毫秒级精准释放）"""
        metrics = self._client_metrics[client_id]
        elapsed = metrics.record_inflight_end()
        # 如果释放时间异常短（<100ms），可能是错误路径，记录日志
        if elapsed > 0 and elapsed < 0.1:
            logger.debug(f"快速释放请求: client={client_id[:8]} elapsed={elapsed:.3f}s")


    # ========== v9.2 并发统计（供用户控制台显示） ==========


    # ========== 并发限制（已移除：v11.0 解除限制释放性能） ==========


    # ========== 评分公式 ==========

    def calculate_score(self, key_id: str, model: str, base_weight: float) -> float:
        """
        计算桶的最终得分

        最终得分 = 基础权重 × 5xx退避 × 隔离 × 健康度 × 预热 × 趋势 × 弹性 × 轮询 各乘数
        """
        bucket = self._get_bucket(key_id, model)

        # 5xx退避乘数
        m_5xx = self.get_5xx_multiplier(bucket)
        # 隔离乘数
        m_iso = self.get_isolation_multiplier(bucket)
        # 健康度权重乘数
        m_health = self.get_base_weight_multiplier(bucket)
        # 预热乘数
        m_warmup = self.get_warmup_multiplier(bucket)
        # 趋势感知乘数
        m_trae = self.get_trae_multiplier(key_id, model)
        # 弹性调度乘数
        m_lobster = self.get_lobster_multiplier(key_id)
        # 轮询均衡乘数（Round-Robin Balanced Dispatch）
        m_rr = self.get_round_robin_multiplier(key_id, model)

        score = base_weight * m_5xx * m_iso * m_health * m_warmup * m_trae * m_lobster * m_rr
        return score

    # ========== select_key 核心方法（严格公平调度） ==========

    def select_key(self, model: str) -> Optional[Tuple[str, str, str, str, str]]:
        """
        核心调度方法 - 严格公平调度

        评分仅用于健康性过滤，最终选择由公平排序决定

        【强制约束】
        1. 检查冷却状态时只使用 (key_id, model) 复合键查询 bucket.cooldown_until
        2. 不存在任何以单一密钥查询冷却状态的代码路径
        3. 软繁忙（运行时集合）与冷却（cooldown_until字段）完全分离

        返回: (key_id, api_key, model, provider, base_url)
        - provider 字符串如 "nvidia"/"openai"/"zhipu"/"aigcly"（用于选择上游适配器）
        - base_url 该 key 自定义上游地址; 空串表示用适配器默认
        """
        self._stats.total_select_calls += 1

        # 模型级429熔断检查：如果某模型短时间内429太多，直接返回429给客户端
        # 避免反复尝试→反复429→反复降级的恶性循环
        now = time.time()
        circuit_breaker_until = self._model_429_circuit_breaker.get(model, 0)
        if now < circuit_breaker_until:
            # 熔断中，直接返回None（调用方会返回429给客户端）
            self._stats.total_none_returns += 1
            return None

        # 模型级5xx熔断检查：如果某模型短时间内5xx太多，暂停调度避免浪费所有密钥
        circuit_breaker_5xx_until = self._model_5xx_circuit_breaker.get(model, 0)
        if now < circuit_breaker_5xx_until:
            logger.warning(f"模型级5xx熔断: model={model} 持续至{circuit_breaker_5xx_until:.0f}")
            self._stats.total_none_returns += 1
            return None

        # 获取所有活跃上游密钥
        upstreams = self._get_active_keys()
        if not upstreams:
            logger.warning("select_key: 无活跃上游密钥")
            self._stats.total_none_returns += 1
            return None

        # ===== 多 Provider 路由：按 model→providers 映射过滤候选 =====
        # 规范格式为 {model_id: [provider, ...]}；读取时兼容旧的单 provider 字符串。
        # 模型存在映射但没有任何 active 上游时，必须返回无候选，不能误发到无关上游。
        if upstreams:
            # 聚合模型的链路项可使用 provider::model 形式，精确锁定一个上游。
            # 未带前缀的历史模型 ID 仍按旧映射在多个供应商间路由，保持兼容。
            selected_provider = ""
            if "::" in model:
                candidate_provider, candidate_model = model.split("::", 1)
                if candidate_provider and candidate_model:
                    selected_provider = candidate_provider

            if selected_provider:
                upstreams = [
                    upstream for upstream in upstreams
                    if upstream.get("provider") == selected_provider
                ]
                logger.info(
                    f"select_key: model={model} 精确路由 provider={selected_provider}, "
                    f"候选数={len(upstreams)}"
                )
                if not upstreams:
                    self._stats.total_none_returns += 1
                    return None
                mapping = {}
            else:
                try:
                    mapping_json = get_setting("model_to_provider") or "{}"
                    import json as _json
                    mapping = _json.loads(mapping_json) if mapping_json else {}
                except Exception:
                    mapping = {}
            if not selected_provider and isinstance(mapping, dict) and model in mapping:
                raw_targets = mapping[model]
                target_providers = raw_targets if isinstance(raw_targets, list) else [raw_targets]
                target_providers = {
                    provider for provider in target_providers
                    if isinstance(provider, str) and provider
                }
                # 兼容曾写入过的 key_id 值：只在当前 active 上游中解析，
                # 解析不到则保持无候选，绝不回退到无关上游。
                provider_by_key_id = {
                    upstream.get("id"): upstream.get("provider")
                    for upstream in upstreams
                    if upstream.get("id") and upstream.get("provider")
                }
                target_providers = {
                    provider_by_key_id.get(provider, provider)
                    for provider in target_providers
                }
                upstreams = [
                    upstream for upstream in upstreams
                    if upstream.get("provider") in target_providers
                ]
                logger.info(
                    f"select_key: model={model} 命中映射 providers={sorted(target_providers)}, "
                    f"候选数={len(upstreams)}"
                )
                if not upstreams:
                    logger.warning(
                        f"select_key: model={model} 已配置路由但没有活跃候选 "
                        f"providers={sorted(target_providers)}"
                    )
                    self._stats.total_none_returns += 1
                    return None

        now = time.time()

        # === 第一轮：硬性过滤（不可放宽） ===
        # 5元组: (key_id, api_key, provider, base_url, filtered_reason)
        hard_filtered = []
        for up in upstreams:
            key_id = up["id"]
            provider = up.get("provider", "nvidia")
            base_url = up.get("base_url") or ""
            bucket = self._get_bucket(key_id, model)

            # 冷却状态过滤
            if bucket.is_cooled_down():
                hard_filtered.append((key_id, "", provider, base_url, "cooldown"))
                continue
            # 隔离过滤
            if bucket.is_isolated():
                hard_filtered.append((key_id, "", provider, base_url, "isolation"))
                continue
            # 自愈过滤
            if bucket.is_healing():
                hard_filtered.append((key_id, "", provider, base_url, "healing"))
                continue
            # 密钥解密
            api_key = self._ensure_key_cached(key_id, up["api_key_ciphertext"])
            if not api_key:
                hard_filtered.append((key_id, "", provider, base_url, "decrypt_failed"))
                continue

            # v10.0: 自动停用密钥过滤（连续认证失败被停用的密钥）
            if key_id in self._auto_deactivated_keys:
                hard_filtered.append((key_id, "", provider, base_url, "auto_deactivated"))
                continue

            hard_filtered.append((key_id, api_key, provider, base_url, "ok"))

        # 收集通过硬性过滤的密钥
        eligible = [(kid, ak, prov, burl) for kid, ak, prov, burl, reason in hard_filtered if reason == "ok"]
        if not eligible:
            # 降级模式：所有密钥都在冷却/隔离中，选择冷却剩余时间最短的桶强制恢复
            best_cooldown_key = None
            best_cooldown_provider = "nvidia"
            best_cooldown_base_url = ""
            min_remaining = float('inf')
            for key_id, _, provider, base_url, reason in hard_filtered:
                if reason == "cooldown":
                    bucket = self._get_bucket(key_id, model)
                    remaining = bucket.cooldown_until - now
                    if remaining < min_remaining:
                        min_remaining = remaining
                        best_cooldown_key = key_id
                        best_cooldown_provider = provider
                        best_cooldown_base_url = base_url
                elif reason in ("healing", "isolation", "decrypt_failed"):
                    # 隔离/自愈/解密失败的桶不能强制恢复
                    continue

            if best_cooldown_key:
                # 强制解除该桶的冷却，让它立即参与轮询
                bucket = self._get_bucket(best_cooldown_key, model)
                bucket.cooldown_until = 0  # 立即恢复
                bucket.warmup_progress = 0
                api_key = self._ensure_key_cached(best_cooldown_key, None)
                if api_key:
                    logger.warning(
                        f"select_key降级模式: 所有密钥冷却中，强制恢复 key={best_cooldown_key[:8]} "
                        f"model={model} 剩余冷却={min_remaining:.1f}s"
                    )
                    self.record_key_usage(best_cooldown_key, model)
                    return (best_cooldown_key, api_key, model, best_cooldown_provider, best_cooldown_base_url)

            # 真正无法选择（无密钥或所有密钥解密失败）
            logger.warning(f"select_key: 无法选择密钥 model={model} (候选池为空，降级模式也无可用密钥)")
            self._stats.total_none_returns += 1
            return None

        # === 第二轮：软性过滤 + 健康性评分 ===
        # 密钥不放弃，软过滤仅用于排序优先级调整
        # 8元组: (bucket_count, key_count, dispatch_seq, score, key_id, api_key, provider, base_url)
        healthy_candidates = []
        soft_filtered = []      # 被软性过滤的候选（作为备选）

        for key_id, api_key, provider, base_url in eligible:
            bucket_key = (key_id, model)
            base_weight = float(self._get_key_weight(key_id, 1.0))
            score = self.calculate_score(key_id, model, base_weight)

            # 计算公平调度参数
            bucket_count, key_count = self.get_fair_dispatch_count(key_id, model)
            dispatch_seq = self._bucket_dispatch_seq.get(bucket_key, 0)

            # 密钥都参与轮询，不因软繁忙/预判/低评分排除
            healthy_candidates.append((bucket_count, key_count, dispatch_seq, score, key_id, api_key, provider, base_url))

        # 密钥都参与轮询，不再需要最小候选池保障

        if not healthy_candidates:
            # 已在eligible阶段处理降级模式，此处理论上不应到达
            logger.warning(f"select_key: 健康候选池为空（不应到达此处） model={model}")
            self._stats.total_none_returns += 1
            return None

        # === 严格公平排序 ===
        # 排序优先级（5密钥池适配）：
        # 1. 桶级近期使用次数（升序）- 使用最少的优先
        # 2. 分配序号（升序）- 同计数时按轮询顺序（确保严格轮询）
        # 3. 健康评分（降序）- 最终质量保障
        # 注：移除key_count作为排序键，避免跨模型计数干扰单模型均衡
        healthy_candidates.sort(key=lambda x: (x[0], x[2], -x[3]))

        best = healthy_candidates[0]
        bucket_count, key_count, dispatch_seq, score, key_id, api_key, provider, base_url = best

        # 记录密钥使用（驱动公平调度）
        self.record_key_usage(key_id, model)

        logger.info(
            f"select_key选中: key={key_id[:8]} model={model} provider={provider} "
            f"bucket_used={bucket_count} key_used={key_count} "
            f"score={score:.4f} (候选{len(healthy_candidates)}个)"
        )

        return (key_id, api_key, model, provider, base_url)

    def _get_key_weight(self, key_id: str, default: float = 1.0) -> float:
        """从缓存的活跃密钥中获取指定密钥的权重"""
        for up in self._active_keys_cache:
            if up["id"] == key_id:
                return float(up.get("weight", default))
        return default


    # ========== 状态查询方法（管理后台用） ==========

    def get_bucket_stats(self) -> List[dict]:
        """获取所有桶的状态快照（管理后台用）"""
        now = time.time()
        result = []
        for (key_id, model), bucket in self._buckets.items():
            result.append({
                "key_id": key_id,
                "model": model,
                "rpm": bucket.get_rpm(),
                "threshold": get_threshold_for_model(model, bucket.dynamic_threshold),
                "success_rate": round(bucket.get_success_rate(), 2),
                "avg_rt": round(bucket.get_avg_rt(), 3),
                "p95_rt": round(bucket.get_p95_rt(), 3),
                "total_requests": bucket.total_requests,
                "total_success": bucket.total_success,
                "total_failures": bucket.total_failures,
                "total_429": bucket.total_429,
                "total_5xx": bucket.total_5xx,
                "total_timeout": bucket.total_timeout,
                "cooldown_remaining": max(0, int(bucket.cooldown_until - now)) if bucket.is_cooled_down() else 0,
                "isolation_remaining": max(0, int(bucket.isolation_until - now)) if bucket.is_isolated() else 0,
                "health_score": round(bucket.health_score, 1),
                "warmup_progress": bucket.warmup_progress,
                "consecutive_5xx": bucket.consecutive_5xx,
                "consecutive_conn_fail": bucket.consecutive_conn_fail,
                "soft_busy": (key_id, model) in self._soft_busy,
                "predicted_busy": (key_id, model) in self._predicted_busy,
                "heal_action": bucket.heal_action,
                "last_success_at": bucket.last_success_at,
                "last_failure_at": bucket.last_failure_at,
                "last_failure_type": bucket.last_failure_type,
            })
        return result


    def get_global_status(self) -> dict:
        """获取全局状态总览（统一字段名，提供前端期望的所有指标）"""
        total_buckets = len(self._buckets)
        cooled_buckets = sum(1 for b in self._buckets.values() if b.is_cooled_down())
        isolated_buckets = sum(1 for b in self._buckets.values() if b.is_isolated())
        soft_busy_buckets = len(self._soft_busy)
        warmup_buckets = sum(1 for b in self._buckets.values() if 0 < b.warmup_progress < self.WARMUP_TARGET)

        # 平均健康分
        avg_health = 100.0
        try:
            health_scores = [b.health_score for b in self._buckets.values()]
            if health_scores:
                avg_health = sum(health_scores) / len(health_scores)
        except Exception:
            logger.exception(f"[v10.0] 调度器后台任务异常")

        # 修复：当后台任务未运行时，动态计算健康密钥数
        # 避免服务刚启动时healthy_key_count=0但实际所有桶都健康的情况
        healthy_key_count = self._healthy_key_count
        if healthy_key_count == 0 and total_buckets > 0:
            # 后台任务尚未运行，动态计算
            healthy_keys = set()
            for (key_id, model), bucket in self._buckets.items():
                if bucket.health_score >= 50 and not bucket.is_cooled_down() and not bucket.is_isolated():
                    healthy_keys.add(key_id)
            healthy_key_count = len(healthy_keys)

        # 当前RPM (最近60秒请求数近似)
        current_rpm = 0
        try:
            now = time.time()
            for b in self._buckets.values():
                current_rpm += b.get_rpm()
        except Exception:
            logger.exception(f"[v10.0] 调度器后台任务异常")

        # 在途请求数
        # v9.2: 在途请求数 - 使用毫秒级精确计数（自动清理超时请求后）
        inflight_requests = 0
        try:
            for m in self._client_metrics.values():
                inflight_requests += m.get_real_inflight_count()
        except Exception:
            inflight_requests = sum(m.inflight_count for m in self._client_metrics.values())

        # 当前QPS (RPM/60)
        current_qps = current_rpm / 60.0 if current_rpm > 0 else 0.0

        # 活跃密钥数（数据库中active状态的密钥）
        try:
            active_key_count = len(fetch_all("SELECT id FROM upstream_keys WHERE status='active'"))
        except Exception:
            active_key_count = healthy_key_count

        # 修复：降级模式判断也要考虑动态计算的healthy_key_count
        degraded_mode = self._stats.degraded_mode
        if not degraded_mode and healthy_key_count < 3 and total_buckets > 0:
            # 仅当有桶数据但健康密钥不足3个时才标记降级
            # 避免无数据时误报降级
            pass

        return {
            # 系统状态
            "degraded_mode": degraded_mode,
            # 密钥
            "healthy_key_count": healthy_key_count,
            "active_key_count": active_key_count,
            "upstream_keys": active_key_count,  # 别名
            # 桶
            "total_buckets": total_buckets,
            "cooling_buckets": cooled_buckets,
            "cooled_buckets": cooled_buckets,  # 别名（向后兼容）
            "isolated_buckets": isolated_buckets,
            "soft_busy_buckets": soft_busy_buckets,
            "warmup_buckets": warmup_buckets,
            # 实时流量
            "current_qps": round(current_qps, 2),
            "current_rpm": current_rpm,
            "inflight_requests": inflight_requests,
            # 调度器调用统计
            "select_calls": self._stats.total_select_calls,
            "total_select_calls": self._stats.total_select_calls,  # 别名
            "total_none_returns": self._stats.total_none_returns,
            # 健康度
            "avg_health_score": round(avg_health, 2),
        }


    # ========== 后台清理任务 ==========


    async def _cleanup_invalid_model_buckets(self):
        """只清理本地无效桶，不再用某个 Provider 的模型列表删除其它 Provider 的统计。"""
        now = time.time()
        last = getattr(self, "_last_invalid_cleanup", 0.0)
        if now - last < 300.0:
            return
        self._last_invalid_cleanup = now

        empty_keys = [(key_id, model) for key_id, model in self._buckets if not model]
        for key in empty_keys:
            self._buckets.pop(key, None)
            self._soft_busy.discard(key)
            self._predicted_busy.discard(key)
            self._bucket_recent_usage.pop(key, None)
            self._bucket_last_used.pop(key, None)
            self._bucket_dispatch_seq.pop(key, None)
        if empty_keys:
            logger.info(f"清理空模型桶: 删除{len(empty_keys)}个")

    # ========== 主动密钥健康检查 ==========

    async def _proactive_key_health_check(self):
        """按每个密钥自己的 Provider 和 Base URL 做无副作用凭据探测。"""
        now = time.time()
        last = getattr(self, "_last_health_check", 0.0)
        if now - last < 30.0:
            return
        self._last_health_check = now

        active_keys = self._get_active_keys()
        if not active_keys:
            return
        check_idx = getattr(self, "_health_check_idx", 0)
        if check_idx >= len(active_keys):
            check_idx = 0
        batch = active_keys[check_idx:check_idx + 5]
        self._health_check_idx = check_idx + 5

        from app.provider_probe import ProviderProbe, probe_provider

        for key_row in batch:
            key_id = key_row["id"]
            api_key = self._ensure_key_cached(key_id, key_row["api_key_ciphertext"])
            if not api_key:
                logger.debug(f"密钥健康检查跳过: key={key_id[:8]} 无法解密")
                continue
            result = await probe_provider(ProviderProbe(
                provider=key_row.get("provider", ""),
                base_url=key_row.get("base_url", ""),
                api_key=api_key,
            ))
            if result.get("ok"):
                logger.debug(f"密钥健康检查通过: key={key_id[:8]} provider={key_row.get('provider', '')}")
            else:
                logger.info(
                    f"密钥健康检查失败（不自动停用）: key={key_id[:8]} "
                    f"provider={key_row.get('provider', '')} reason={result.get('reason', 'unknown')}"
                )

    # ========== 后台任务入口 ==========

    async def run_background_tasks(self):
        """运行所有后台周期任务（含自适应资源感知调度）"""
        # 系统资源检查（内存压力感知）
        await self._check_system_resources()

        # 核心调度算法（始终执行）
        # 每30秒更新自适应阈值
        self.update_adaptive_thresholds()
        # 每30秒更新健康度
        self.update_health_scores()
        # 更新负载预判
        self.update_load_prediction()
        
        self.run_self_heal()
        
        self.run_lobster_molting()

        # 内存感知的清理任务调度
        is_high_pressure = self._memory_pressure > 0.7
        is_critical = self._memory_pressure >= 1.0

        # 低压力下执行完整清理；高压力下跳过非关键清理
        if not is_high_pressure:
            # 清理过期客户端治理数据（防内存泄漏）
            self._cleanup_stale_clients()
            # 清理软繁忙集合中已恢复的桶
            self._cleanup_soft_busy()
            # 清理滑动窗口中的过期时间戳（防内存增长）
            self._cleanup_fair_dispatch_windows()
            # 安全修复：清理过期的密钥明文缓存
            self._cleanup_expired_key_cache()
        elif is_critical:
            # 临界压力下立即触发日志清理释放磁盘空间
            logger.warning("系统内存临界，触发紧急日志清理")
            self._run_log_cleanup()

        # 效模型桶（模型不存在于上游NVIDIA NIM）
        await self._cleanup_invalid_model_buckets()

        # v10.0：主动密钥健康检查（每30秒测试一批密钥）
        await self._proactive_key_health_check()

        # 数据库日志自动清理（每6小时执行一次，内存节约模式每1小时）
        self._cleanup_counter += 1
        effective_interval = self.CLEANUP_INTERVAL_CYCLES // 6 if self._memory_saving_mode else self.CLEANUP_INTERVAL_CYCLES
        if self._cleanup_counter >= effective_interval:
            self._cleanup_counter = 0
            self._run_log_cleanup()


    # ========== 客户端密钥缓存 ==========

    def get_client_key_cache(self, key_hash: str) -> Optional[dict]:
        """获取缓存的客户端密钥信息，若过期则返回 None"""
        now = time.time()
        if now - self._client_key_cache_time > self.CLIENT_KEY_CACHE_TTL:
            return None
        return self._client_key_cache.get(key_hash)

    def set_client_key_cache(self, key_hash: str, client_info: dict) -> None:
        """缓存客户端密钥信息并更新时间戳"""
        self._client_key_cache[key_hash] = client_info
        self._client_key_cache_time = time.time()


# ========== 全局调度器实例 ==========

_scheduler: Optional[SurgeScheduler] = None


def get_scheduler() -> SurgeScheduler:
    """获取全局调度器实例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = SurgeScheduler()
    return _scheduler


def reset_scheduler():
    """重置调度器（测试用）"""
    global _scheduler
    _scheduler = None
