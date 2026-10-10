"""
异步数据库访问模块 - 后端切换器

sqlite 模式：所有 async API 代理到同步 database_sqlite，通过 asyncio.to_thread 包装。
postgres 模式：直接 re-export db_async_pool_pg 的 asyncpg 实现。
"""
import os
import logging
import asyncio
from typing import Any, Optional

_BACKEND = (os.environ.get("DATABASE_BACKEND", "sqlite")).lower()
logger = logging.getLogger("acu.async_db")

if _BACKEND == "sqlite":
    # SQLite 模式：同步后端 + to_thread 包装
    from app import database_sqlite as _sqlite

    _pool = None

    async def init_async_pool() -> None:
        logger.info("[async_db] sqlite 模式：跳过 asyncpg 池创建")
        return

    async def close_async_pool() -> None:
        return

    def _ensure_pool():
        return None

    async def fetch_one(sql: str, *params: Any) -> Optional[dict]:
        return await asyncio.to_thread(_sqlite.fetch_one, sql.replace("%s", "?"), tuple(params))

    async def fetch_all(sql: str, *params: Any) -> list:
        return await asyncio.to_thread(_sqlite.fetch_all, sql.replace("%s", "?"), tuple(params))

    async def execute(sql: str, *params: Any) -> str:
        rc = await asyncio.to_thread(_sqlite.execute, sql.replace("%s", "?"), tuple(params))
        return str(rc)

    async def execute_many(sql: str, params_list: list) -> None:
        adapted = sql.replace("%s", "?")
        for params in params_list:
            await asyncio.to_thread(_sqlite.execute, adapted, tuple(params))

    async def execute_batch(sql_params_list: list) -> None:
        for sql, params in sql_params_list:
            await asyncio.to_thread(_sqlite.execute, sql.replace("%s", "?"), tuple(params))

    logger.info("[async_db] 后端 = sqlite")
else:
    # PostgreSQL 模式：直接透传原版 asyncpg 实现
    from app.db_async_pool_pg import (  # noqa: F401
        init_async_pool, close_async_pool, fetch_one, fetch_all,
        execute, execute_many, execute_batch,
    )
    logger.info("[async_db] 后端 = postgres (asyncpg)")
