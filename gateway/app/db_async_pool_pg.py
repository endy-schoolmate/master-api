"""PostgreSQL 原版 asyncpg 实现，仅在 DATABASE_BACKEND=postgres 下由 db_async_pool 导入并 re-export。"""

"""
异步数据库访问模块 - asyncpg 连接池

提供基于 asyncpg 的连接池管理，所有数据库操作均通过连接池执行。
包含查询超时、连接健康检查等机制，以替代同步的 psycopg2 直连方式。
"""
import os
import logging
import asyncio
from typing import Any, Optional

import asyncpg

logger = logging.getLogger("acu.async_db")

# 数据库连接配置（从环境变量读取）
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DB = os.environ.get("PG_DB", "master_gateway")
PG_USER = os.environ.get("PG_USER", "master")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

# 连接池配置
POOL_MIN_SIZE = 5          # 池中最小连接数
POOL_MAX_SIZE = 15         # 池中最大连接数（pool_size + max_overflow = 5 + 10）
QUERY_TIMEOUT = 10.0       # 查询超时秒数

_pool: Optional[asyncpg.Pool] = None


async def _on_connect_hook(conn: asyncpg.Connection) -> None:
    """连接建立时的健康检查钩子

    每个新创建的连接都会执行此钩子，确保连接可用。
    """
    try:
        await conn.execute("SELECT 1")
        logger.debug("异步连接健康检查通过")
    except Exception as e:
        logger.error(f"异步连接健康检查失败: {e}")
        raise


async def init_async_pool() -> None:
    """创建异步连接池

    使用 asyncpg.create_pool 创建带健康检查连接的连接池。
    幂等操作：如果池已存在则跳过。
    """
    global _pool
    if _pool is not None:
        logger.warning("异步连接池已存在，跳过创建")
        return

    try:
        _pool = await asyncpg.create_pool(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            init=_on_connect_hook,
            # 连接空闲超时后自动回收（秒）
            max_inactive_connection_lifetime=300.0,
        )
        logger.info(
            f"异步连接池创建成功: host={PG_HOST}:{PG_PORT}, db={PG_DB}, "
            f"pool_size={POOL_MIN_SIZE}/{POOL_MAX_SIZE}"
        )
    except (OSError, asyncpg.PostgresError) as e:
        logger.error(f"创建异步连接池失败: {e}")
        raise


async def close_async_pool() -> None:
    """关闭异步连接池

    安全关闭池及其所有连接。幂等操作。
    """
    global _pool
    if _pool is None:
        logger.debug("异步连接池已关闭或未初始化")
        return

    try:
        await _pool.close()
        logger.info("异步连接池已关闭")
    except Exception as e:
        logger.error(f"关闭异步连接池失败: {e}")
        raise
    finally:
        _pool = None


def _ensure_pool() -> asyncpg.Pool:
    """确保连接池已初始化，否则抛出异常"""
    if _pool is None:
        raise RuntimeError(
            "异步连接池未初始化，请先调用 init_async_pool()"
        )
    return _pool


async def fetch_one(sql: str, *params: Any) -> Optional[dict]:
    """查询单条记录，返回 dict 或 None

    Args:
        sql: SQL 查询语句
        *params: 查询参数

    Returns:
        单行结果字典，若无记录返回 None
    """
    pool = _ensure_pool()
    try:
        async with pool.acquire() as conn:
            row = await asyncio.wait_for(
                conn.fetchrow(sql, *params),
                timeout=QUERY_TIMEOUT,
            )
            return dict(row) if row else None
    except asyncio.TimeoutError:
        logger.error(f"查询超时 ({QUERY_TIMEOUT}s): {sql[:80]}...")
        raise
    except asyncpg.PostgresError as e:
        logger.error(f"数据库查询错误: {e}, SQL: {sql[:80]}...")
        raise


async def fetch_all(sql: str, *params: Any) -> list[dict]:
    """查询多条记录，返回 list[dict]

    Args:
        sql: SQL 查询语句
        *params: 查询参数

    Returns:
        多行结果字典列表
    """
    pool = _ensure_pool()
    try:
        async with pool.acquire() as conn:
            rows = await asyncio.wait_for(
                conn.fetch(sql, *params),
                timeout=QUERY_TIMEOUT,
            )
            return [dict(r) for r in rows]
    except asyncio.TimeoutError:
        logger.error(f"查询超时 ({QUERY_TIMEOUT}s): {sql[:80]}...")
        raise
    except asyncpg.PostgresError as e:
        logger.error(f"数据库查询错误: {e}, SQL: {sql[:80]}...")
        raise


async def execute(sql: str, *params: Any) -> str:
    """执行写操作，返回受影响的行数（asyncpg 返回行数字符串）

    Args:
        sql: SQL 写入语句
        *params: 写入参数

    Returns:
        asyncpg 返回的状态字符串（通常包含行数信息）
    """
    pool = _ensure_pool()
    try:
        async with pool.acquire() as conn:
            result = await asyncio.wait_for(
                conn.execute(sql, *params),
                timeout=QUERY_TIMEOUT,
            )
            return result
    except asyncio.TimeoutError:
        logger.error(f"写入超时 ({QUERY_TIMEOUT}s): {sql[:80]}...")
        raise
    except asyncpg.PostgresError as e:
        logger.error(f"数据库写入错误: {e}, SQL: {sql[:80]}...")
        raise


async def execute_many(sql: str, params_list: list[tuple]) -> None:
    """批量执行相同的写入语句

    Args:
        sql: SQL 写入语句
        params_list: 参数元组列表，每个元组对应一次执行
    """
    pool = _ensure_pool()
    try:
        async with pool.acquire() as conn:
            await asyncio.wait_for(
                conn.executemany(sql, params_list),
                timeout=QUERY_TIMEOUT,
            )
    except asyncio.TimeoutError:
        logger.error(f"批量写入超时 ({QUERY_TIMEOUT}s): {sql[:80]}...")
        raise
    except asyncpg.PostgresError as e:
        logger.error(f"数据库批量写入错误: {e}, SQL: {sql[:80]}...")
        raise


async def execute_batch(sql_params_list: list[tuple[str, tuple]]) -> None:
    """在一个事务中执行多条不同的 SQL 语句

    适用于批量日志写入等需要事务一致性的场景。
    所有 SQL 在同一个事务中执行，任一失败则全部回滚。

    Args:
        sql_params_list: (sql, params) 元组列表
    """
    pool = _ensure_pool()
    if not sql_params_list:
        return

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                for sql, params in sql_params_list:
                    await asyncio.wait_for(
                        conn.execute(sql, *params),
                        timeout=QUERY_TIMEOUT,
                    )
    except asyncio.TimeoutError:
        logger.error(f"事务批量写入超时 ({QUERY_TIMEOUT}s)")
        raise
    except asyncpg.PostgresError as e:
        logger.error(f"事务批量写入错误: {e}")
        raise
