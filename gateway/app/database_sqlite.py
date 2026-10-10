"""
SQLite 后端适配层 - 同步接口

提供与 database.py (PostgreSQL) 完全相同的函数签名，
让本项目无需 PostgreSQL 也能在本地单机运行。

设计要点：
- 同进程全局连接 + check_same_thread=False，兼容 scheduler 跨线程同步调用
- 开启 WAL 模式提升并发读写
- 自动 SQL 转换：%s -> ?, SERIAL -> INTEGER PRIMARY KEY AUTOINCREMENT,
  ON CONFLICT (key) DO UPDATE SET ... -> INSERT OR REPLACE（仅对 admin_settings 等单 PK 表）,
  其余按需手动适配
- 异步路径(db_async_pool)也代理到这里，避免依赖 asyncpg

SQL 方言差异处理策略：
- fetch_one/fetch_all/execute 收到 SQL 时统一做轻量字符串替换：
    * `%s` -> `?`          （参数占位符，所有 PG SQL 都用 %s）
    * `~*` -> `LIKE`       （不区分大小写正则 → 用 LIKE，足够本地用）
    * `ILIKE` -> `LIKE`
- ON CONFLICT 这种 PG 特有的 upsert 在本层用专门函数处理，
  不走通用 execute 转换（避免误伤）
"""
from __future__ import annotations

import os
import sqlite3
import uuid
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

logger = logging.getLogger("acu.database.sqlite")

# ============ 时区 (与 database.py 完全一致) ============

CST = timezone(timedelta(hours=8))

DB_PATH = os.environ.get("MASTER_API_SQLITE_PATH", "master.db")

# ============ 全局连接 ============

_conn: Optional[sqlite3.Connection] = None
_conn_lock = threading.RLock()

def _get_conn() -> sqlite3.Connection:
    """获取全局 SQLite 连接(线程安全)。"""
    global _conn
    if _conn is None:
        with _conn_lock:
            if _conn is None:
                conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
                # autocommit 模式(isolation_level=None)：每个 execute 自成事务
                conn.row_factory = sqlite3.Row
                # 启用 WAL 提升并发
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA foreign_keys=ON")
                _conn = conn
                logger.info(f"[SQLite] 已连接: {DB_PATH} (WAL)")
    return _conn

def _put_conn(conn) -> None:
    """SQLite 单进程单连接, 无需归还 - 占位以与 PG 版签名一致"""
    pass

# ============ SQL 占位符与轻型方言转换 ============

def _adapt_sql(sql: str) -> str:
    """把 PostgreSQL 风格 SQL 转成 SQLite 可执行的形式。

    只做最保守的字符串替换:
    - `%s` -> `?`
    - `ILIKE` -> `LIKE`, `~*` -> `LIKE` (本地自用足够)
    - 大小写无关的 `SERIAL PRIMARY KEY` -> `INTEGER PRIMARY KEY AUTOINCREMENT`
      （建表时用。注意：本转换只在 init_db 的 CREATE TABLE 上下文里需要,
        普通 SELECT/INSERT 不会触发 SERIAL 关键字, 所以共用此函数无副作用）
    """
    if sql is None:
        return sql
    # 参数占位符
    sql = sql.replace("%s", "?")
    # 不区分大小写的模糊匹配
    sql = sql.replace("ILIKE", "LIKE")
    sql = sql.replace("~*", "LIKE")
    # SERIAL：实际只在 CREATE TABLE 出现, 安全全局替换
    sql = sql.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
    sql = sql.replace("SERIAL", "INTEGER PRIMARY KEY AUTOINCREMENT")
    return sql

# ============ 时间工具（与 database.py 行为一致）============

def utcnow() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

def localnow() -> str:
    now = datetime.now(CST)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}+08:00"

def localnow_ms() -> int:
    return int(datetime.now(CST).timestamp() * 1000)

def today_start_utc() -> str:
    now_cst = datetime.now(CST)
    midnight_cst = now_cst.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight_cst.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

def today_start_local() -> str:
    now_cst = datetime.now(CST)
    midnight_cst = now_cst.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight_cst.strftime("%Y-%m-%dT%H:%M:%S.000+08:00")

def days_ago_utc(days: int) -> str:
    now_cst = datetime.now(CST)
    target = (now_cst - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    return target.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

# ============ 核心读写接口 ============

def execute(sql: str, params: tuple = ()) -> int:
    """执行写操作并提交, 返回受影响行数。"""
    conn = _get_conn()
    with _conn_lock:
        cur = conn.execute(_adapt_sql(sql), tuple(params))
        # autocommit 模式下已自动提交
        return cur.rowcount

def fetch_one(sql: str, params: tuple = ()) -> Optional[dict]:
    conn = _get_conn()
    with _conn_lock:
        cur = conn.execute(_adapt_sql(sql), tuple(params))
        row = cur.fetchone()
        return dict(row) if row else None

def fetch_all(sql: str, params: tuple = ()) -> list:
    conn = _get_conn()
    with _conn_lock:
        cur = conn.execute(_adapt_sql(sql), tuple(params))
        return [dict(r) for r in cur.fetchall()]

# ============ admin_settings upsert（SQLite 专属）============

def get_setting(key: str) -> Optional[str]:
    row = fetch_one("SELECT value FROM admin_settings WHERE key = ?", (key,))
    return row["value"] if row else None

def set_setting(key: str, value: str) -> None:
    """SQLite 用 INSERT OR REPLACE 实现 upsert。"""
    now = utcnow()
    execute(
        "INSERT OR REPLACE INTO admin_settings (key, value, updated_at) VALUES (?, ?, ?)",
        (key, value, now),
    )

def insert_audit(action: str, target_type: str, target_id: str, detail: str) -> None:
    execute(
        "INSERT INTO audit_logs (id, operator, action, target_type, target_id, detail, created_at) "
        "VALUES (?, 'admin', ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), action, target_type, target_id, detail, utcnow()),
    )

# ============ 建表 / 迁移 ============

def init_db() -> None:
    """创建所有表 (SQLite 语法, 与 PG 版表语义一致)。"""
    conn = _get_conn()
    with _conn_lock:
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
                last_health_check_latency_ms INTEGER,
                last_health_check_error TEXT DEFAULT '',
                weight INTEGER DEFAULT 1,
                rpm_limit INTEGER DEFAULT 40,
                switch_threshold INTEGER DEFAULT 38,
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
                status_code INTEGER,
                latency_ms INTEGER,
                retried INTEGER DEFAULT 0,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                is_stream INTEGER DEFAULT 0,
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
        try:
            log_cols = [r[1] for r in cur.execute("PRAGMA table_info(request_logs)").fetchall()]
            for col_name, col_def in [
                ("cached_tokens", "INTEGER DEFAULT 0"),
                ("cache_write_tokens", "INTEGER DEFAULT 0"),
                ("cost", "REAL DEFAULT 0.0"),
                ("currency", "TEXT DEFAULT 'USD'"),
            ]:
                if col_name not in log_cols:
                    cur.execute(f"ALTER TABLE request_logs ADD COLUMN {col_name} {col_def};")
        except Exception as e:
            logger.warning(f"Error checking/migrating request_logs columns: {e}")

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
                total_requests INTEGER DEFAULT 0,
                total_success INTEGER DEFAULT 0,
                total_failures INTEGER DEFAULT 0,
                consecutive_failures INTEGER DEFAULT 0,
                total_429 INTEGER DEFAULT 0,
                total_5xx INTEGER DEFAULT 0,
                total_timeout INTEGER DEFAULT 0,
                daily_requests INTEGER DEFAULT 0,
                daily_success INTEGER DEFAULT 0,
                daily_failures INTEGER DEFAULT 0,
                daily_date TEXT,
                weekly_requests INTEGER DEFAULT 0,
                weekly_success INTEGER DEFAULT 0,
                weekly_date TEXT,
                monthly_requests INTEGER DEFAULT 0,
                monthly_success INTEGER DEFAULT 0,
                monthly_date TEXT,
                avg_rt REAL DEFAULT 0,
                p95_rt REAL DEFAULT 0,
                last_success_at TEXT,
                last_failure_at TEXT,
                last_failure_type TEXT,
                updated_at TEXT
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '新对话',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT DEFAULT '',
                reasoning TEXT DEFAULT '',
                attachments TEXT DEFAULT '[]',
                model TEXT DEFAULT '',
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions(updated_at);")

        # 迁移字段
        _migrate_request_logs_columns(conn)
        _migrate_clients_user_type(conn)
        _migrate_upstream_keys_base_url(conn)
        _migrate_upstream_keys_health_check(conn)
        _migrate_upstream_keys_auth_type(conn)
        logger.info("[SQLite] init_db 完成")

def _get_column_names(conn, table_name: str) -> set:
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    return {row["name"] for row in cur.fetchall()}

def _migrate_request_logs_columns(conn) -> None:
    """合并原 PG 的两段迁移(精度+全量)，一次补齐所有字段。"""
    existing = _get_column_names(conn, "request_logs")
    needed = [
        ("started_at", "TEXT DEFAULT ''"),
        ("completed_at", "TEXT DEFAULT ''"),
        ("latency_us", "INTEGER DEFAULT 0"),
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
        ("gateway_dispatch_ms", "REAL DEFAULT 0"),
    ]
    for col, col_def in needed:
        if col not in existing:
            conn.execute(f"ALTER TABLE request_logs ADD COLUMN {col} {col_def}")
    for idx_sql in (
        "CREATE INDEX IF NOT EXISTS idx_request_logs_path ON request_logs(request_path)",
        "CREATE INDEX IF NOT EXISTS idx_request_logs_status ON request_logs(status_code)",
        "CREATE INDEX IF NOT EXISTS idx_request_logs_category ON request_logs(log_category)",
        "CREATE INDEX IF NOT EXISTS idx_request_logs_client_ip ON request_logs(client_ip)",
        "CREATE INDEX IF NOT EXISTS idx_request_logs_status_created ON request_logs(status_code, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_request_logs_model_created ON request_logs(model, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_request_logs_client_status_created ON request_logs(client_id, status_code, created_at)",
    ):
        conn.execute(idx_sql)

def _migrate_clients_user_type(conn) -> None:
    existing = _get_column_names(conn, "clients")
    if "user_type" not in existing:
        conn.execute("ALTER TABLE clients ADD COLUMN user_type TEXT DEFAULT 'old'")
        logger.info("[SQLite] clients 增加 user_type")

def _migrate_upstream_keys_base_url(conn) -> None:
    """为 upstream_keys 加 base_url 字段（多 provider 改造核心）。"""
    existing = _get_column_names(conn, "upstream_keys")
    if "base_url" not in existing:
        conn.execute("ALTER TABLE upstream_keys ADD COLUMN base_url TEXT DEFAULT ''")
        logger.info("[SQLite] upstream_keys 增加 base_url")

# ============ 默认配置种子 ============

def _migrate_upstream_keys_health_check(conn) -> None:
    """Add persisted admin connectivity check results for existing databases."""
    existing = _get_column_names(conn, "upstream_keys")
    needed = [
        ("last_health_check_status", "TEXT DEFAULT 'unknown'"),
        ("last_health_check_at", "TEXT DEFAULT ''"),
        ("last_health_check_latency_ms", "INTEGER"),
        ("last_health_check_error", "TEXT DEFAULT ''"),
    ]
    for col, col_def in needed:
        if col not in existing:
            conn.execute(f"ALTER TABLE upstream_keys ADD COLUMN {col} {col_def}")

def _migrate_upstream_keys_auth_type(conn) -> None:
    """为 upstream_keys 加 auth_type 字段（api_key | oauth，ChatGPT 账号登录）。"""
    existing = _get_column_names(conn, "upstream_keys")
    if "auth_type" not in existing:
        conn.execute("ALTER TABLE upstream_keys ADD COLUMN auth_type TEXT DEFAULT 'api_key'")
        logger.info("[SQLite] upstream_keys 增加 auth_type")

def seed_defaults() -> None:
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

    from app.security import (
        generate_upstream_master_key, generate_platform_key, hash_secret,
    )
    if get_setting("upstream_master_key") is None:
        set_setting("upstream_master_key", generate_upstream_master_key())
    if get_setting("gateway_secret") is None:
        set_setting("gateway_secret", generate_platform_key())

# ============ 日志清理（SQLite 版）============

def cleanup_success_logs(keep_days: int = 3, keep_error_days: int = 90) -> dict:
    result = {"success_deleted": 0, "error_deleted": 0}
    try:
        success_cutoff = days_ago_utc(keep_days)
        row = fetch_one(
            "SELECT COUNT(*) as cnt FROM request_logs WHERE status_code = 200 AND created_at < ?",
            (success_cutoff,),
        )
        success_count = row["cnt"] if row else 0
        if success_count > 0:
            execute(
                "DELETE FROM request_logs WHERE status_code = 200 AND created_at < ?",
                (success_cutoff,),
            )
            result["success_deleted"] = success_count

        error_cutoff = days_ago_utc(keep_error_days)
        row = fetch_one(
            "SELECT COUNT(*) as cnt FROM request_logs WHERE status_code != 200 AND created_at < ?",
            (error_cutoff,),
        )
        error_count = row["cnt"] if row else 0
        if error_count > 0:
            execute(
                "DELETE FROM request_logs WHERE status_code != 200 AND created_at < ?",
                (error_cutoff,),
            )
            result["error_deleted"] = error_count

        # SQLite 无需 VACUUM request_logs; 偶尔 VACUUM 全库降碎片
        try:
            conn = _get_conn()
            conn.execute("PRAGMA optimize")
        except Exception:
            pass

        if success_count > 0 or error_count > 0:
            logger.info(
                f"[SQLite] 日志清理: 删除{success_count}条成功日志(>{keep_days}天), "
                f"{error_count}条错误日志(>{keep_error_days}天)"
            )
    except Exception as e:
        logger.error(f"[SQLite] 日志清理失败: {e}")
    return result
