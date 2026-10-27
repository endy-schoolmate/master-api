"""
网关数据库模块 - PostgreSQL (psycopg2) 同步访问

v7.0 关键变更：
- upstream_keys 表移除所有冷却状态字段（cooling_until/current_minute_requests等）
  冷却状态完全由调度器的桶级字典管理，不持久化到DB
- request_logs 表包含完整token统计字段
"""
import os
import sys
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

logger = logging.getLogger("acu.database")

class _SqliteModeActivated(Exception):
    """SQLite 模式下用于短路 database.py 剩余 PG 专属代码的执行。"""

# ===== 后端切换 =====
# 默认 sqlite 单机本地部署。如需 PostgreSQL，在 .env 设 DATABASE_BACKEND=postgres
_BACKEND = (os.environ.get("DATABASE_BACKEND", "sqlite")).lower()

if _BACKEND == "sqlite":
    from app import database_sqlite as _sqlite
    sys.modules[__name__] = _sqlite
    logger.info("[database] 后端 = sqlite (本地单机模式)")
else:
        # ===== PostgreSQL 原路径 =====
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool

    # 异步连接池支持（可选，需要 asyncpg）
    from app import db_async_pool

    # 连接池（全局单例）
    _pool = None

    def _init_pool():
        """初始化连接池"""
        global _pool
        if _pool is None:
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=5, maxconn=30,
                host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASSWORD,
            )

    # 数据库连接配置（从环境变量读取）
    PG_HOST = os.environ.get("PG_HOST", "localhost")
    PG_PORT = int(os.environ.get("PG_PORT", "5432"))
    PG_DB = os.environ.get("PG_DB", "master_gateway")
    PG_USER = os.environ.get("PG_USER", "master")
    PG_PASSWORD = os.environ.get("PG_PASSWORD")
    if not PG_PASSWORD:
        raise RuntimeError("[FATAL] 环境变量 PG_PASSWORD 未设置！请设置后重新启动。")

    # 中国标准时区 UTC+8
    CST = timezone(timedelta(hours=8))

    def _get_conn():
        """从连接池获取 PostgreSQL 连接"""
        if _pool is None:
            _init_pool()
        conn = _pool.getconn()
        conn.autocommit = False
        # 使用 RealDictCursor 使返回行为 dict
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        return conn

    def _put_conn(conn):
        """归还连接到连接池"""
        if _pool and conn:
            _pool.putconn(conn)

    def utcnow() -> str:
        """返回ISO格式UTC时间（毫秒精度）"""
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    def localnow() -> str:
        """返回本地时间(CST+8)ISO格式（毫秒精度）"""
        now = datetime.now(CST)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}+08:00"

    def localnow_ms() -> int:
        """返回当前本地时间的毫秒级时间戳"""
        return int(datetime.now(CST).timestamp() * 1000)

    def today_start_utc() -> str:
        """返回今日本地零点(CST)对应的UTC时间（用于数据库查询边界）

        例如：本地时间 2026-07-16 14:30 CST
        返回：2026-07-15T16:00:00.000Z (即 CST 2026-07-16 00:00:00)
        """
        now_cst = datetime.now(CST)
        midnight_cst = now_cst.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight_cst.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def today_start_local() -> str:
        """返回今日本地零点的ISO格式（CST）"""
        now_cst = datetime.now(CST)
        midnight_cst = now_cst.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight_cst.strftime("%Y-%m-%dT%H:%M:%S.000+08:00")

    def days_ago_utc(days: int) -> str:
        """返回N天前本地零点(CST)对应的UTC时间"""
        now_cst = datetime.now(CST)
        target = (now_cst - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        return target.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def execute(sql: str, params: tuple = ()) -> int:
        """执行写操作并提交，返回受影响的行数"""
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount
        finally:
            _put_conn(conn)

    def fetch_one(sql: str, params: tuple = ()) -> Optional[dict]:
        """查询单条记录"""
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            _put_conn(conn)

    def fetch_all(sql: str, params: tuple = ()) -> list:
        """查询多条记录"""
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
        finally:
            _put_conn(conn)

    # ─── 异步连接池桥接函数 ─────────────────────────────
    async def async_get_pool():
        """返回 db_async_pool 模块，用于获取异步连接池及其函数

        用法：
            pool_mod = await async_get_pool()
            row = await pool_mod.fetch_one("SELECT * FROM ...")
            rows = await pool_mod.fetch_all("SELECT * FROM ...")
            result = await pool_mod.execute("INSERT INTO ...", val1, val2)

        首次调用时会自动初始化连接池。
        """
        if db_async_pool._pool is None:
            await db_async_pool.init_async_pool()
        return db_async_pool

    def get_setting(key: str) -> Optional[str]:
        """获取配置项"""
        row = fetch_one("SELECT value FROM admin_settings WHERE key = %s", (key,))
        return row["value"] if row else None

    def set_setting(key: str, value: str) -> None:
        """设置配置项"""
        now = utcnow()
        execute(
            "INSERT INTO admin_settings (key, value, updated_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = %s",
            (key, value, now, value, now),
        )

    def insert_audit(action: str, target_type: str, target_id: str, detail: str) -> None:
        """插入审计日志"""
        execute(
            "INSERT INTO audit_logs (id, operator, action, target_type, target_id, detail, created_at) "
            "VALUES (%s, 'admin', %s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), action, target_type, target_id, detail, utcnow()),
        )

    def init_db() -> None:
        """创建所有表（v10.0 完整表结构，PostgreSQL 语法）"""
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS upstream_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    provider TEXT DEFAULT 'nvidia',
                    api_key_ciphertext TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                base_url TEXT DEFAULT '',
                last_health_check_status TEXT DEFAULT 'unknown',
                last_health_check_at TEXT DEFAULT '',
                last_health_check_latency_ms INT,
                last_health_check_error TEXT DEFAULT '',
                weight INT DEFAULT 1,
                    rpm_limit INT DEFAULT 40,
                    switch_threshold INT DEFAULT 38,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clients (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS client_api_keys (
                    id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    key_prefix TEXT NOT NULL,
                    key_ciphertext TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    FOREIGN KEY (client_id) REFERENCES clients(id)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id TEXT PRIMARY KEY,
                    client_id TEXT,
                    upstream_key_id TEXT,
                    model TEXT,
                    status_code INT,
                    latency_ms INT,
                    ttft_ms REAL DEFAULT 0,
                    total_duration_ms REAL DEFAULT 0,
                    retried INT DEFAULT 0,
                    prompt_tokens INT DEFAULT 0,
                    completion_tokens INT DEFAULT 0,
                    total_tokens INT DEFAULT 0,
                    is_stream INT DEFAULT 0,
                    error_msg TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_client_created ON request_logs(client_id, created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_model ON request_logs(model);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_pricing (
                    id TEXT PRIMARY KEY,
                    model_id TEXT UNIQUE NOT NULL,
                    display_name TEXT DEFAULT '',
                    provider TEXT DEFAULT '',
                    currency TEXT DEFAULT 'USD',
                    input_cost REAL DEFAULT 0,
                    output_cost REAL DEFAULT 0,
                    cache_read_cost REAL DEFAULT 0,
                    cache_write_cost REAL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    operator TEXT,
                    action TEXT,
                    target_type TEXT,
                    target_id TEXT,
                    detail TEXT,
                    created_at TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS key_usage_stats (
                    key_id TEXT PRIMARY KEY,
                    total_requests INT DEFAULT 0,
                    total_success INT DEFAULT 0,
                    total_failures INT DEFAULT 0,
                    consecutive_failures INT DEFAULT 0,
                    total_429 INT DEFAULT 0,
                    total_5xx INT DEFAULT 0,
                    total_timeout INT DEFAULT 0,
                    daily_requests INT DEFAULT 0,
                    daily_success INT DEFAULT 0,
                    daily_failures INT DEFAULT 0,
                    daily_date TEXT,
                    weekly_requests INT DEFAULT 0,
                    weekly_success INT DEFAULT 0,
                    weekly_date TEXT,
                    monthly_requests INT DEFAULT 0,
                    monthly_success INT DEFAULT 0,
                    monthly_date TEXT,
                    avg_rt REAL DEFAULT 0,
                    p95_rt REAL DEFAULT 0,
                    last_success_at TEXT,
                    last_failure_at TEXT,
                    last_failure_type TEXT,
                    updated_at TEXT
                );
            """)
            conn.commit()

            # 迁移：为 request_logs 添加毫秒级精准统计字段
            _migrate_request_logs_precision(conn)
            #  迁移：为 request_logs 添加全量请求日志字段
            _migrate_request_logs_full(conn)
            # v9.2 迁移：为 clients 表添加 user_type 字段
            _migrate_clients_user_type(conn)
            _migrate_upstream_keys_health_check(conn)
        finally:
            # v10.0 修复：归还连接到连接池而非关闭（防止连接池耗尽）
            _put_conn(conn)

    def _get_column_names(conn, table_name: str) -> set:
        """查询指定表的现有列名（PostgreSQL information_schema）"""
        cur = conn.cursor()
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s AND table_schema = 'public'",
            (table_name,),
        )
        return {row["column_name"] for row in cur.fetchall()}

    def _migrate_request_logs_precision(conn) -> None:
        """迁移：为 request_logs 添加 started_at/completed_at/latency_us 字段"""
        existing_cols = _get_column_names(conn, "request_logs")

        migrations = [
            ("started_at", "TEXT DEFAULT ''"),
            ("completed_at", "TEXT DEFAULT ''"),
            ("latency_us", "BIGINT DEFAULT 0"),  # 微秒级延迟（1ms=1000us）
            ("ttft_ms", "REAL DEFAULT 0"),  # 首字时长（TTFT）
            ("total_duration_ms", "REAL DEFAULT 0"),  # 总使用时长
        ]
        for col_name, col_def in migrations:
            if col_name not in existing_cols:
                cur = conn.cursor()
                cur.execute(f"ALTER TABLE request_logs ADD COLUMN {col_name} {col_def}")
        conn.commit()

    def _migrate_request_logs_full(conn) -> None:
        """迁移：为 request_logs 添加全量请求日志字段"""
        existing_cols = _get_column_names(conn, "request_logs")

        migrations = [
            ("request_path", "TEXT DEFAULT ''"),
            ("http_method", "TEXT DEFAULT ''"),
            ("client_ip", "TEXT DEFAULT ''"),
            ("user_agent", "TEXT DEFAULT ''"),
            ("request_params", "TEXT DEFAULT ''"),
            ("request_body", "TEXT DEFAULT ''"),
            ("response_body", "TEXT DEFAULT ''"),
            ("error_type", "TEXT DEFAULT ''"),
            ("error_detail", "TEXT DEFAULT ''"),
            ("error_stack", "TEXT DEFAULT ''"),
            ("business_code", "TEXT DEFAULT ''"),
            ("log_category", "TEXT DEFAULT 'normal'"),
        ]
        cur = conn.cursor()
        for col_name, col_def in migrations:
            if col_name not in existing_cols:
                cur.execute(f"ALTER TABLE request_logs ADD COLUMN {col_name} {col_def}")

        # 为新字段添加索引
        cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_path ON request_logs(request_path)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_status ON request_logs(status_code)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_category ON request_logs(log_category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_client_ip ON request_logs(client_ip)")
        #  复合索引：优化统计查询性能
        cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_status_created ON request_logs(status_code, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_model_created ON request_logs(model, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_client_status_created ON request_logs(client_id, status_code, created_at)")
        conn.commit()

    def seed_defaults() -> None:
        """插入默认配置"""
        defaults = {
            "upstream_base_url": "https://integrate.api.nvidia.com/v1",
            "chat_path": "/chat/completions",
            "models_path": "/models",
            "cooldown_seconds": "240",
            "switch_threshold": "38",
            "model_mappings": "{}",
        }
        for key, value in defaults.items():
            if get_setting(key) is None:
                set_setting(key, value)

        # 确保主密钥和网关密钥存在
        from app.security import generate_upstream_master_key, generate_platform_key, hash_secret
        if get_setting("upstream_master_key") is None:
            set_setting("upstream_master_key", generate_upstream_master_key())
        if get_setting("gateway_secret") is None:
            set_setting("gateway_secret", generate_platform_key())

    def cleanup_success_logs(keep_days: int = 3, keep_error_days: int = 90) -> dict:
        """
        清理请求日志：成功日志短期保留，错误日志长期保存

        Args:
            keep_days: 成功日志保留天数（默认3天）
            keep_error_days: 错误日志保留天数（默认90天）

        Returns:
            dict: {"success_deleted": 删除的成功日志数, "error_deleted": 删除的错误日志数}
        """
        import logging
        logger = logging.getLogger("acu.database")

        result = {"success_deleted": 0, "error_deleted": 0}

        try:
            # 删除过期的成功日志（status_code=200且超过keep_days天）
            success_cutoff = days_ago_utc(keep_days)
            row = fetch_one(
                "SELECT COUNT(*) as cnt FROM request_logs WHERE status_code = 200 AND created_at < %s",
                (success_cutoff,),
            )
            success_count = row["cnt"] if row else 0
            if success_count > 0:
                execute(
                    "DELETE FROM request_logs WHERE status_code = 200 AND created_at < %s",
                    (success_cutoff,),
                )
                result["success_deleted"] = success_count

            # 删除过期的错误日志（status_code!=200且超过keep_error_days天）
            error_cutoff = days_ago_utc(keep_error_days)
            row = fetch_one(
                "SELECT COUNT(*) as cnt FROM request_logs WHERE status_code != 200 AND created_at < %s",
                (error_cutoff,),
            )
            error_count = row["cnt"] if row else 0
            if error_count > 0:
                execute(
                    "DELETE FROM request_logs WHERE status_code != 200 AND created_at < %s",
                    (error_cutoff,),
                )
                result["error_deleted"] = error_count

            if success_count > 0 or error_count > 0:
                logger.info(f"日志清理: 删除{success_count}条成功日志(>{keep_days}天), {error_count}条错误日志(>{keep_error_days}天)")

            # PostgreSQL VACUUM（需要单独连接且 autocommit=True）
            try:
                conn = psycopg2.connect(
                    host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                    user=PG_USER, password=PG_PASSWORD,
                )
                conn.autocommit = True
                cur = conn.cursor()
                cur.execute("VACUUM request_logs")
                cur.close()
                conn.close()
            except Exception as e:
                logger.warning(f"VACUUM 失败(可忽略): {e}")

        except Exception as e:
            logger.error(f"日志清理失败: {e}")

        return result

    def _migrate_upstream_keys_health_check(conn) -> None:
        """Add persisted admin connectivity check results for existing databases."""
        existing = _get_column_names(conn, "upstream_keys")
        migrations = [
            ("base_url", "TEXT DEFAULT ''"),
            ("last_health_check_status", "TEXT DEFAULT 'unknown'"),
            ("last_health_check_at", "TEXT DEFAULT ''"),
            ("last_health_check_latency_ms", "INT"),
            ("last_health_check_error", "TEXT DEFAULT ''"),
        ]
        cur = conn.cursor()
        for col_name, col_def in migrations:
            if col_name not in existing:
                cur.execute(f"ALTER TABLE upstream_keys ADD COLUMN {col_name} {col_def}")
        conn.commit()

    def _migrate_clients_user_type(conn):
        """v9.2 迁移：为 clients 表添加 user_type 字段（old=老用户4并发, new=新用户2并发）"""
        existing = _get_column_names(conn, "clients")
        if "user_type" not in existing:
            cur = conn.cursor()
            cur.execute("ALTER TABLE clients ADD COLUMN user_type TEXT DEFAULT 'old'")
            conn.commit()
            logger.info("数据库迁移: clients 表已添加 user_type 字段 (默认 'old')")
