"""
网关 SQLAdmin 管理面板

提供数据库表的 Web 管理界面，挂载于 /gw/dbadmin。
使用与 admin_api.py 相同的管理员密码进行认证。
"""
import os

from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from starlette.responses import RedirectResponse
import bcrypt

from app.db_async import async_engine
from app.models import (
    AdminSetting,
    UpstreamKey,
    Client,
    ClientApiKey,
    RequestLog,
    AuditLog,
    KeyUsageStat,
)


# ========== 认证后端 ==========

class AdminAuth(AuthenticationBackend):
    """SQLAdmin 认证：使用 MASTER_ADMIN_PASSWORD 环境变量"""

    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = form.get("username", "")
        password = form.get("password", "")
        admin_password = os.environ.get("MASTER_ADMIN_PASSWORD", "")
        if password == admin_password:
            request.session.update({"authenticated": True})
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get("authenticated", False)


# ========== 模型视图配置 ==========

class AdminSettingView(ModelView, model=AdminSetting):
    name = "管理设置"
    name_plural = "管理设置"
    icon = "fa-solid fa-gear"
    column_list = [AdminSetting.key, AdminSetting.value, AdminSetting.updated_at]
    column_searchable_list = [AdminSetting.key]
    can_delete = True


class UpstreamKeyView(ModelView, model=UpstreamKey):
    name = "上游密钥"
    name_plural = "上游密钥"
    icon = "fa-solid fa-key"
    column_list = [
        UpstreamKey.id, UpstreamKey.name, UpstreamKey.provider,
        UpstreamKey.key_prefix, UpstreamKey.weight, UpstreamKey.rpm_limit,
        UpstreamKey.switch_threshold, UpstreamKey.status,
        UpstreamKey.created_at, UpstreamKey.updated_at,
    ]
    column_searchable_list = [UpstreamKey.name, UpstreamKey.provider]
    column_sortable_list = [UpstreamKey.name, UpstreamKey.weight, UpstreamKey.status]
    form_excluded_columns = [UpstreamKey.api_key_ciphertext]
    can_delete = True


class ClientView(ModelView, model=Client):
    name = "客户端"
    name_plural = "客户端"
    icon = "fa-solid fa-users"
    column_list = [Client.id, Client.name, Client.status, Client.created_at, Client.updated_at]
    column_searchable_list = [Client.name]
    can_delete = True


class ClientApiKeyView(ModelView, model=ClientApiKey):
    name = "客户端API密钥"
    name_plural = "客户端API密钥"
    icon = "fa-solid fa-id-badge"
    column_list = [
        ClientApiKey.id, ClientApiKey.client_id, ClientApiKey.key_prefix,
        ClientApiKey.status, ClientApiKey.created_at, ClientApiKey.last_used_at,
    ]
    column_searchable_list = [ClientApiKey.client_id]
    form_excluded_columns = [ClientApiKey.key_hash, ClientApiKey.key_ciphertext]
    can_delete = True


class RequestLogView(ModelView, model=RequestLog):
    name = "请求日志"
    name_plural = "请求日志"
    icon = "fa-solid fa-list"
    column_list = [
        RequestLog.id, RequestLog.client_id, RequestLog.upstream_key_id,
        RequestLog.model, RequestLog.status_code, RequestLog.latency_ms,
        RequestLog.total_tokens, RequestLog.is_stream, RequestLog.created_at,
    ]
    column_searchable_list = [RequestLog.model, RequestLog.client_id]
    column_sortable_list = [RequestLog.created_at, RequestLog.latency_ms, RequestLog.status_code]
    column_default_sort = ("created_at", True)  # 降序
    can_create = False
    can_edit = False
    can_delete = True


class AuditLogView(ModelView, model=AuditLog):
    name = "审计日志"
    name_plural = "审计日志"
    icon = "fa-solid fa-clipboard"
    column_list = [
        AuditLog.id, AuditLog.operator, AuditLog.action,
        AuditLog.target_type, AuditLog.target_id, AuditLog.created_at,
    ]
    column_searchable_list = [AuditLog.action, AuditLog.operator]
    column_default_sort = ("created_at", True)
    can_create = False
    can_edit = False
    can_delete = True


class KeyUsageStatView(ModelView, model=KeyUsageStat):
    name = "密钥使用统计"
    name_plural = "密钥使用统计"
    icon = "fa-solid fa-chart-bar"
    column_list = [
        KeyUsageStat.key_id, KeyUsageStat.total_requests, KeyUsageStat.total_success,
        KeyUsageStat.total_failures, KeyUsageStat.avg_rt, KeyUsageStat.updated_at,
    ]
    column_searchable_list = [KeyUsageStat.key_id]
    can_create = False
    can_delete = True


# ========== 创建 Admin 实例 ==========

def create_admin(app=None) -> Admin:
    """创建 SQLAdmin 实例并注册所有模型"""
    authentication_backend = AdminAuth(secret_key=os.environ.get("ADMIN_SESSION_SECRET", ""))
    admin = Admin(
        app=app,
        engine=async_engine,
        authentication_backend=authentication_backend,
        title="Master API 网关数据库管理",
        base_url="/gw/dbadmin",
    )

    admin.add_view(AdminSettingView)
    admin.add_view(UpstreamKeyView)
    admin.add_view(ClientView)
    admin.add_view(ClientApiKeyView)
    admin.add_view(RequestLogView)
    admin.add_view(AuditLogView)
    admin.add_view(KeyUsageStatView)

    return admin
