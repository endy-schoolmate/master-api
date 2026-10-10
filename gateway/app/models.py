"""
网关 SQLAlchemy 2.0 异步 ORM 模型

与 database.py 中的表结构完全对应，用于 SQLAdmin 管理面板和未来异步迁移。
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ========== 管理设置 ==========

class AdminSetting(Base):
    __tablename__ = "admin_settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)


# ========== 上游密钥 ==========

class UpstreamKey(Base):
    __tablename__ = "upstream_keys"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str] = mapped_column(String, default="nvidia")
    api_key_ciphertext: Mapped[str] = mapped_column(String, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String, nullable=False)
    base_url: Mapped[str] = mapped_column(String, default="")
    last_health_check_status: Mapped[str] = mapped_column(String, default="unknown")
    last_health_check_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_health_check_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_health_check_error: Mapped[str] = mapped_column(Text, default="")
    weight: Mapped[int] = mapped_column(Integer, default=1)
    rpm_limit: Mapped[int] = mapped_column(Integer, default=40)
    switch_threshold: Mapped[int] = mapped_column(Integer, default=38)
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


# ========== 客户端 ==========

class Client(Base):
    __tablename__ = "clients"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)

    api_keys: Mapped[list["ClientApiKey"]] = relationship(back_populates="client")


# ========== 客户端API密钥 ==========

class ClientApiKey(Base):
    __tablename__ = "client_api_keys"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    client_id: Mapped[str] = mapped_column(String, ForeignKey("clients.id"), nullable=False)
    key_hash: Mapped[str] = mapped_column(String, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String, nullable=False)
    key_ciphertext: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    last_used_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    client: Mapped["Client"] = relationship(back_populates="api_keys")


# ========== 请求日志 ==========

class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    client_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    upstream_key_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    retried: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    is_stream: Mapped[int] = mapped_column(Integer, default=0)
    error_msg: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[str] = mapped_column(String, default="")
    completed_at: Mapped[str] = mapped_column(String, default="")
    latency_us: Mapped[int] = mapped_column(Integer, default=0)
    #  新增：全量请求日志字段
    request_path: Mapped[str] = mapped_column(String, default="")
    http_method: Mapped[str] = mapped_column(String, default="")
    client_ip: Mapped[str] = mapped_column(String, default="")
    user_agent: Mapped[str] = mapped_column(String, default="")
    request_params: Mapped[str] = mapped_column(Text, default="")
    request_body: Mapped[str] = mapped_column(Text, default="")
    response_body: Mapped[str] = mapped_column(Text, default="")
    error_type: Mapped[str] = mapped_column(String, default="")
    error_detail: Mapped[str] = mapped_column(Text, default="")
    error_stack: Mapped[str] = mapped_column(Text, default="")
    business_code: Mapped[str] = mapped_column(String, default="")
    log_category: Mapped[str] = mapped_column(String, default="normal")  # normal / error / auth_fail


# ========== 审计日志 ==========

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    operator: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    action: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    target_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    target_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)


# ========== 密钥使用统计 ==========

class KeyUsageStat(Base):
    __tablename__ = "key_usage_stats"

    key_id: Mapped[str] = mapped_column(String, primary_key=True)
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    total_success: Mapped[int] = mapped_column(Integer, default=0)
    total_failures: Mapped[int] = mapped_column(Integer, default=0)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    total_429: Mapped[int] = mapped_column(Integer, default=0)
    total_5xx: Mapped[int] = mapped_column(Integer, default=0)
    total_timeout: Mapped[int] = mapped_column(Integer, default=0)
    daily_requests: Mapped[int] = mapped_column(Integer, default=0)
    daily_success: Mapped[int] = mapped_column(Integer, default=0)
    daily_failures: Mapped[int] = mapped_column(Integer, default=0)
    daily_date: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    weekly_requests: Mapped[int] = mapped_column(Integer, default=0)
    weekly_success: Mapped[int] = mapped_column(Integer, default=0)
    weekly_date: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    monthly_requests: Mapped[int] = mapped_column(Integer, default=0)
    monthly_success: Mapped[int] = mapped_column(Integer, default=0)
    monthly_date: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    avg_rt: Mapped[float] = mapped_column(Float, default=0)
    p95_rt: Mapped[float] = mapped_column(Float, default=0)
    last_success_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_failure_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_failure_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
