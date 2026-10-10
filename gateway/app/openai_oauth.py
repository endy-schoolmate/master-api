"""Sign in with ChatGPT（OpenAI 账号登录）OAuth 登录流。

公开客户端流程：
- 动态客户端注册：authorize 用 client_id="dynamic_agent_client"，OpenAI 在回调
  query 里返回签发的 client_id，后续 token 交换/刷新都用它；
- PKCE(S256) + state，本地回调服务器 http://127.0.0.1:1455/auth/callback；
- scope 必须含 chatgpt.tokens.use.direct（direct token sharing）；
- 拿到的 access token 直调 api.openai.com 的 Responses API（订阅计费）；
- 支持手动粘贴回调 URL 兜底（回调端口被占用/浏览器不通时）。

凭据以 JSON 信封形式加密存进 upstream_keys.api_key_ciphertext（auth_type='oauth'），
转发前由 ensure_fresh_access() 换取/刷新 access token 并回写数据库。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import time
import uuid

import httpx

from app.database import execute, get_setting, set_setting, utcnow
from app.security import encrypt_upstream_key, mask_secret

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME_HINT = "Master API"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
DIRECT_TOKEN_SCOPE = "chatgpt.tokens.use.direct"
SCOPE = f"openid profile email offline_access resource.invoke {DIRECT_TOKEN_SCOPE}"
EXPIRY_MARGIN_S = 180
PENDING_TTL_S = 900  # 待完成登录会话 15 分钟有效

ENVELOPE_PREFIX = '{"_oauth'

# 访问 OpenAI（登录/token/对话转发）的可选代理：admin_settings.oauth_proxy
# 或环境变量 MASTER_API_OAUTH_PROXY（如 http://127.0.0.1:7890）。
# restart.bat 会清掉系统代理环境变量，直连被 OpenAI 地区封锁时需配置此项。
_PROXY_CLIENT: httpx.AsyncClient | None = None
_PROXY_CLIENT_PROXY: str | None = None

# OAuth 调试日志（排查 invalid_grant 等问题用）：gateway/logs/oauth_debug.log
_DEBUG_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "logs", "oauth_debug.log")
_TOKEN_REDACT = re.compile(r'("(?:access_token|refresh_token|id_token)"\s*:\s*")[^"]+(")')


def _dlog(msg: str) -> None:
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG), exist_ok=True)
        from datetime import datetime
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


class OAuthError(Exception):
    """OAuth 流程错误（面向管理界面的中文消息）。"""


def get_oauth_proxy() -> str | None:
    proxy = ""
    try:
        proxy = get_setting("oauth_proxy") or ""
    except Exception:
        proxy = ""
    return (proxy or os.environ.get("MASTER_API_OAUTH_PROXY") or "").strip() or None


async def get_proxied_client() -> httpx.AsyncClient | None:
    """转发 openai-codex 上游用的代理长连接池（未配置代理返回 None，走通用池）。"""
    global _PROXY_CLIENT
    proxy = get_oauth_proxy()
    if not proxy:
        return None
    global _PROXY_CLIENT_PROXY
    if _PROXY_CLIENT is not None and _PROXY_CLIENT_PROXY == proxy and not _PROXY_CLIENT.is_closed:
        return _PROXY_CLIENT
    if _PROXY_CLIENT is not None and not _PROXY_CLIENT.is_closed:
        try:
            await _PROXY_CLIENT.aclose()
        except Exception:
            pass
    _PROXY_CLIENT = httpx.AsyncClient(
        proxy=proxy,
        timeout=httpx.Timeout(600.0, connect=30.0),
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=16, keepalive_expiry=120),
    )
    _PROXY_CLIENT_PROXY = proxy
    return _PROXY_CLIENT


# --------------------------------------------------------------------------
# 凭据信封（加密前的明文形态）

def make_envelope(cred: dict) -> str:
    return json.dumps({"_oauth": {"v": 1, **cred}}, ensure_ascii=False)


def parse_envelope(plaintext: str) -> dict | None:
    """解出 OAuth 凭据；非信封（普通 API Key）返回 None。"""
    if not plaintext or not plaintext.lstrip().startswith(ENVELOPE_PREFIX):
        return None
    try:
        data = json.loads(plaintext)
    except Exception:
        return None
    cred = (data or {}).get("_oauth")
    if not isinstance(cred, dict) or not cred.get("access") or not cred.get("refresh"):
        return None
    return cred


def envelope_for_storage(cred: dict, master_key: str) -> tuple[str, str]:
    """返回 (密文, 脱敏前缀)。"""
    plaintext = make_envelope(cred)
    return encrypt_upstream_key(plaintext, master_key), mask_secret(cred.get("access", ""))


def redacted_envelope(plaintext: str) -> dict:
    """reveal 用：展示凭据概要，refresh_token 打码。"""
    cred = parse_envelope(plaintext)
    if cred is None:
        return {}
    shown = dict(cred)
    refresh = shown.get("refresh", "")
    if len(refresh) > 12:
        shown["refresh"] = refresh[:6] + "***" + refresh[-4:]
    return shown


# --------------------------------------------------------------------------
# PKCE / 设备标识

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _agent_host_id() -> str:
    """OpenAI 用稳定 UUID 标识本机这个 agent 宿主（urn:uuid:<uuid>）。"""
    device_id = ""
    try:
        device_id = get_setting("oauth_agent_host_id") or ""
    except Exception:
        device_id = ""
    if not device_id:
        device_id = str(uuid.uuid4())
        try:
            set_setting("oauth_agent_host_id", device_id)
        except Exception:
            pass
    return f"urn:uuid:{device_id.lower()}"


# --------------------------------------------------------------------------
# 待完成登录会话 + 本地回调服务器

class _LoginSession:
    def __init__(self, state_id: str, state: str, verifier: str, login_url: str, callback_ok: bool):
        self.state_id = state_id
        self.state = state
        self.verifier = verifier
        self.login_url = login_url
        self.callback_ok = callback_ok
        self.status = "pending"          # pending | completed | failed
        self.error = ""
        self.cred: dict | None = None
        self.created_at = time.time()

    def expired(self) -> bool:
        return time.time() - self.created_at > PENDING_TTL_S


_sessions: dict[str, _LoginSession] = {}
_callback_server: asyncio.AbstractServer | None = None
_callback_sessions: dict[str, _LoginSession] = {}  # state → session（仅回调服务器存活期）


def get_session(state_id: str) -> _LoginSession | None:
    sess = _sessions.get(state_id)
    if sess and sess.expired():
        if sess.status == "pending":
            sess.status = "failed"
            sess.error = "登录会话已过期，请重新发起"
        else:
            _sessions.pop(state_id, None)
            return None
    return sess


def pop_completed_session(state_id: str) -> _LoginSession:
    """创建上游密钥时消费一个已完成登录的会话。"""
    sess = get_session(state_id)
    if sess is None:
        raise OAuthError("登录会话不存在或已过期，请重新发起登录")
    if sess.status == "pending":
        raise OAuthError("该登录尚未完成，请先在浏览器完成 ChatGPT 授权")
    if sess.status != "completed" or not sess.cred:
        raise OAuthError(f"登录未成功：{sess.error or '未知错误'}，请重新发起")
    _sessions.pop(state_id, None)
    return sess


def get_completed_access(state_id: str) -> str:
    """读取已完成会话的 access token（不消费会话，供保存前临时测试用）。"""
    sess = get_session(state_id)
    if sess is None or sess.status != "completed" or not sess.cred:
        return ""
    return sess.cred.get("access", "")


def _cleanup_sessions():
    for sid in [sid for sid, s in _sessions.items() if s.expired()]:
        sess = _sessions.pop(sid, None)
        if sess:
            _callback_sessions.pop(sess.state, None)


async def _handle_callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """接收 OpenAI 重定向回 127.0.0.1:1455 的浏览器请求。

    在响应浏览器之前完成凭据交换，页面直接显示真实结果——避免"浏览器显示
    登录成功、后台交换其实失败"的误导（交换失败的真实原因会同时展示出来）。
    """
    try:
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
    except Exception:
        writer.close()
        return
    try:
        first_line = raw.split(b"\r\n", 1)[0].decode("latin-1")
        path = first_line.split(" ")[1] if len(first_line.split(" ")) >= 2 else ""
    except Exception:
        path = ""

    from urllib.parse import parse_qs, urlparse
    from html import escape as html_escape

    status, html, matched = 400, "<h2>Invalid callback</h2>", None
    parsed = urlparse(path if path.startswith("/") else f"/{path}")
    if parsed.path == CALLBACK_PATH:
        qs = parse_qs(parsed.query)
        _dlog(f"callback hit: RAW={path}")
        _dlog(f"callback parsed: query_keys={sorted(qs.keys())} "
              f"code={((qs.get('code') or [''])[0])[:10]}… state={((qs.get('state') or [''])[0])[:8]} "
              f"client_id={((qs.get('client_id') or [''])[0])[:20]} scope={((qs.get('scope') or [''])[0])[:80]} "
              f"error={qs.get('error')}")
        if qs.get("error"):
            status, html = 400, f"<h2>ChatGPT 授权失败</h2><p>{qs['error'][0]}</p>"
        else:
            state = (qs.get("state") or [""])[0]
            sess = _callback_sessions.get(state)
            if sess is None or sess.status != "pending":
                status, html = 400, "<h2>OAuth state 不匹配（可能登录已超时或已完成）</h2>"
            else:
                code = (qs.get("code") or [""])[0]
                client_id = ((qs.get("client_id") or [""])[0]).strip()
                if not code or not client_id:
                    status, html = 400, "<h2>回调缺少 code 或 client_id</h2>"
                else:
                    matched = (sess, code, client_id)

    if matched:
        sess, code, client_id = matched
        # 等交换完成再回页面：成功/失败都展示真实结果
        await _complete_session(sess, code, sess.verifier, client_id)
        if sess.status == "completed":
            status, html = 200, ("<h2>ChatGPT 登录成功</h2>"
                                 "<p>凭据已获取并保存，可关闭此窗口，回到控制台完成添加。</p>")
        else:
            status, html = 502, (
                f"<h2>凭据交换失败</h2><p>{html_escape(sess.error or '未知错误')}</p>"
                "<p>请回到控制台查看状态并重新发起；若为地区/网络限制，请检查网关的 oauth_proxy 代理配置。</p>"
            )

    body = html.encode("utf-8")
    writer.write(
        f"HTTP/1.1 {status} OK\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("latin-1") + body
    )
    try:
        await writer.drain()
    finally:
        writer.close()
    if matched:
        await _stop_callback_server()


async def _start_callback_server() -> bool:
    """绑定 1455 回调端口；被占用（其他会话/Codex CLI）返回 False 走手动粘贴。"""
    global _callback_server
    if _callback_server is not None and _callback_server.is_serving():
        return True
    try:
        _callback_server = await asyncio.start_server(_handle_callback, CALLBACK_HOST, CALLBACK_PORT)
        return True
    except OSError:
        _callback_server = None
        return False


async def _stop_callback_server():
    global _callback_server
    if _callback_server is not None:
        try:
            _callback_server.close()
            await _callback_server.wait_closed()
        except Exception:
            pass
        _callback_server = None
        _callback_sessions.clear()


async def start_login() -> dict:
    """发起一次登录：建会话、起回调服务器、返回授权 URL。"""
    _cleanup_sessions()
    verifier, challenge = _generate_pkce()
    state = _b64url(secrets.token_bytes(32))
    nonce = _b64url(secrets.token_bytes(32))
    callback_ok = await _start_callback_server()

    from urllib.parse import urlencode

    login_url = AUTHORIZE_URL + "?" + urlencode({
        "client_id": DYNAMIC_CLIENT_ID,
        "agent_name_hint": AGENT_NAME_HINT,
        "ext_agent_host_id": _agent_host_id(),
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "resource": RESOURCE,
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "nonce": nonce,
    })
    state_id = uuid.uuid4().hex
    sess = _LoginSession(state_id, state, verifier, login_url, callback_ok)
    _sessions[state_id] = sess
    if callback_ok:
        _callback_sessions[state] = sess
    _dlog(f"login started: state_id={state_id[:8]} state={state[:8]} callback_ok={callback_ok} "
          f"proxy={get_oauth_proxy() or 'DIRECT'} challenge={challenge[:10]}… AUTHORIZE_URL={login_url}")
    return {"state_id": state_id, "login_url": login_url, "callback_ok": callback_ok,
            "redirect_uri": REDIRECT_URI, "expires_in": PENDING_TTL_S}


async def complete_with_paste(state_id: str, pasted_url: str) -> dict:
    """手动粘贴浏览器最终回调 URL 完成登录（回调服务器不可用时的兜底）。"""
    sess = get_session(state_id)
    if sess is None:
        raise OAuthError("登录会话不存在或已过期，请重新发起登录")
    if sess.status != "pending":
        raise OAuthError("该登录会话已结束，请查看状态或重新发起")
    from urllib.parse import urlparse, parse_qs

    url = (pasted_url or "").strip()
    _dlog(f"paste complete: state_id={state_id[:8]} url={url[:70]}")
    parsed = urlparse(url)
    if parsed.scheme != "http" or f"{parsed.hostname}:{parsed.port}" != f"{CALLBACK_HOST}:{CALLBACK_PORT}" \
            or parsed.path != CALLBACK_PATH:
        raise OAuthError(f"粘贴的必须是浏览器最终跳转的完整 URL（以 {REDIRECT_URI} 开头）")
    qs = parse_qs(parsed.query)
    if qs.get("error"):
        raise OAuthError(f"ChatGPT 授权失败：{qs['error'][0]}")
    code = (qs.get("code") or [""])[0]
    client_id = ((qs.get("client_id") or [""])[0]).strip()
    state = (qs.get("state") or [""])[0]
    if not code or not client_id:
        raise OAuthError("回调 URL 中缺少 code 或 client_id（请粘贴完整地址）")
    if state != sess.state:
        raise OAuthError("OAuth state 不匹配，请重新发起登录")
    await _complete_session(sess, code, sess.verifier, client_id)
    return session_status(state_id)


async def _complete_session(sess: _LoginSession, code: str, verifier: str, client_id: str):
    if sess.status == "completed" and sess.cred:
        # 已成功过（浏览器重复回调/刷新）：跳过，避免用已消费的 code 二次交换覆盖成功状态
        _dlog(f"exchange skipped: session state={sess.state[:8]} already completed")
        return
    _dlog(f"exchange start: state={sess.state[:8]} code={code[:10]}… verifier={verifier[:8]}… client_id={client_id[:20]}")
    try:
        sess.cred = await exchange_code(code, verifier, client_id)
        sess.status = "completed"
        _dlog(f"exchange OK: state={sess.state[:8]} access={sess.cred['access'][:8]}… expires_at={sess.cred['expires_at']:.0f}")
    except Exception as exc:
        sess.status = "failed"
        sess.error = str(exc)
        # 失败时记录完整 PKCE 三元组便于本地复核（code 一次性且已失效，无泄露风险）
        _dlog(f"exchange FAILED: state={sess.state[:8]} error={str(exc)[:200]} "
              f"FULL code={code} verifier={verifier} client_id={client_id}")


def session_status(state_id: str) -> dict:
    sess = get_session(state_id)
    if sess is None:
        return {"status": "expired", "error": "登录会话不存在或已过期"}
    data = {"status": sess.status, "callback_ok": sess.callback_ok, "login_url": sess.login_url}
    if sess.error:
        data["error"] = sess.error
    if sess.status == "completed" and sess.cred:
        data["account_hint"] = mask_secret(sess.cred.get("access", ""))
        data["expires_at"] = sess.cred.get("expires_at")
    return data


# --------------------------------------------------------------------------
# token 交换 / 刷新

async def _request_token(form: dict) -> dict:
    proxy = get_oauth_proxy()
    # 记录本机经同一代理访问 OpenAI 的出口 IP（排查授权码 IP 绑定问题）
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), proxy=proxy) as c:
            ip_resp = await c.get("https://api.ipify.org?format=json")
            egress = ip_resp.json().get("ip", "?")
    except Exception as exc:
        egress = f"查询失败({type(exc).__name__})"
    _dlog(f"token request: grant_type={form.get('grant_type')} client_id={str(form.get('client_id'))[:20]} "
          f"code={str(form.get('code'))[:10]}… proxy={proxy or 'DIRECT'} egress_ip={egress}")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0), proxy=proxy) as client:
            resp = await client.post(
                TOKEN_URL,
                headers={"accept": "application/json",
                         "content-type": "application/x-www-form-urlencoded"},
                data=form,
            )
    except httpx.HTTPError as exc:
        _dlog(f"token request transport error: {type(exc).__name__}: {str(exc)[:200]}")
        raise OAuthError(f"连接 OpenAI 授权服务失败：{exc}") from exc
    body = _TOKEN_REDACT.sub(r"\1[REDACTED]\2", resp.text[:500])
    _dlog(f"token response: HTTP {resp.status_code} body={body}")
    if resp.status_code != 200:
        detail = resp.text[:300]
        if resp.status_code == 400 and "invalid_grant" in resp.text:
            detail += ("（invalid_grant 常见原因：① ChatGPT 账号无 Plus/Pro 订阅——本接入方式仅限订阅账号；"
                       "② 授权码过期或被重复使用，请重新发起登录）")
        raise OAuthError(f"OpenAI OAuth token 请求失败（HTTP {resp.status_code}）：{detail}")
    try:
        data = resp.json()
    except Exception as exc:
        raise OAuthError("OpenAI OAuth token 响应不是 JSON") from exc
    if not isinstance(data, dict):
        raise OAuthError("OpenAI OAuth token 响应格式异常")
    return data


async def exchange_code(code: str, verifier: str, client_id: str) -> dict:
    """授权码换凭据；校验 direct-token scope 与 id_token。"""
    token = await _request_token({
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
        "resource": RESOURCE,
    })
    access = token.get("access_token")
    refresh = token.get("refresh_token")
    scope = token.get("scope")
    expires_in = token.get("expires_in")
    if not isinstance(access, str) or not access.strip():
        raise OAuthError("OpenAI OAuth 响应缺少 access_token")
    if not isinstance(refresh, str) or not refresh.strip():
        raise OAuthError("OpenAI OAuth 响应缺少 refresh_token")
    if not isinstance(scope, str) or DIRECT_TOKEN_SCOPE not in scope.split():
        raise OAuthError(f"授权未包含 {DIRECT_TOKEN_SCOPE} scope（账号类型不支持直连），请确认使用 ChatGPT 订阅账号登录")
    if not isinstance(token.get("id_token"), str) or not token["id_token"].strip():
        raise OAuthError("OpenAI OAuth 响应缺少 id_token")
    if not isinstance(expires_in, (int, float)) or expires_in <= 0:
        raise OAuthError("OpenAI OAuth 响应 expires_in 异常")
    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "expires_at": time.time() + float(expires_in) - EXPIRY_MARGIN_S,
        "client_id": client_id,
        "scopes": [s for s in scope.split() if s],
    }


async def refresh_credential(cred: dict) -> dict:
    """用 refresh_token 换新凭据（client_id 必须沿用签发值）。"""
    client_id = (cred.get("client_id") or "").strip()
    if not client_id:
        raise OAuthError("凭据缺少签发的 client_id，请重新登录 ChatGPT")
    token = await _request_token({
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": cred["refresh"],
        "resource": RESOURCE,
    })
    access = token.get("access_token")
    refresh = token.get("refresh_token") or cred["refresh"]
    scope = token.get("scope") or " ".join(cred.get("scopes", []))
    expires_in = token.get("expires_in")
    if not isinstance(access, str) or not access.strip():
        raise OAuthError("OpenAI OAuth 刷新响应缺少 access_token")
    if not isinstance(expires_in, (int, float)) or expires_in <= 0:
        raise OAuthError("OpenAI OAuth 刷新响应 expires_in 异常")
    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "expires_at": time.time() + float(expires_in) - EXPIRY_MARGIN_S,
        "client_id": client_id,
        "scopes": [s for s in str(scope).split() if s] or cred.get("scopes", []),
    }


async def ensure_fresh_access(key_id: str, plaintext: str) -> str:
    """从信封取可用 access token；临近过期自动刷新并回写数据库。

    返回 access token；非 OAuth 信封返回 ""（由调用方决定报错方式）。
    """
    cred = parse_envelope(plaintext)
    if cred is None:
        return ""
    if time.time() < float(cred.get("expires_at") or 0):
        return cred["access"]
    _dlog(f"token refresh: key={key_id[:8]} expired_at={cred.get('expires_at'):.0f}")
    new_cred = await refresh_credential(cred)
    master_key = get_setting("upstream_master_key")
    if master_key:
        ciphertext, key_prefix = envelope_for_storage(new_cred, master_key)
        execute(
            "UPDATE upstream_keys SET api_key_ciphertext = %s, key_prefix = %s, updated_at = %s WHERE id = %s",
            (ciphertext, key_prefix, utcnow(), key_id),
        )
        try:
            from app.scheduler import get_scheduler
            get_scheduler().invalidate_key_cache(key_id)
        except Exception:
            pass
    return new_cred["access"]
