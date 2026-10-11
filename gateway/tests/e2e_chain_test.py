# -*- coding: utf-8 -*-
"""
聚合模型 Failover 端到端自测（真实 HTTP，全链路）

启动一个真实网关实例（独立端口 + 独立临时 sqlite）和一个模拟上游，
模拟上游按"模型名"表现不同故障行为，用真实 HTTP 请求验证这一轮的全部修复：

  T1  200+SSE流内 Validation 参数错误 → 剥离参数原地重试 → 客户端成功（无感）
  T1b 黑名单跨请求生效：第二次调用不再产生 badparam 失败行
  T2  纯 tool_calls 流（Agent 典型回合）→ 不再被误判空流，200 正常收尾
  T3  候选 200 后挂起不出内容 → 探测超时快速切换下一个候选
  T4  全链失败 → 状态码按最后错误类别映射（5xx→502），不再是恒 503
  T5  候选尝试行带实际请求体（脱敏，能看到参数是否被剥离）
  T6  非流式 200+error 体形式的参数错误 → 同样剥离重试成功

运行：cd gateway && ../venv/Scripts/python.exe tests/e2e_chain_test.py
"""
import json
import os
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

# ---- 环境必须在导入 app 之前设置 ----
ROOT = Path(__file__).resolve().parent.parent.parent
TMPDIR = Path(tempfile.mkdtemp(prefix="masterapi-e2e-"))
os.environ["DATABASE_BACKEND"] = "sqlite"
os.environ["MASTER_API_SQLITE_PATH"] = str(TMPDIR / "e2e.db")
# 加速链路：候选预算4s / 整链12s / 探测1.5s / 软熔断2~4s
os.environ["MASTER_API_CHAIN_CANDIDATE_TIMEOUT"] = "4"
os.environ["MASTER_API_CHAIN_TOTAL_BUDGET"] = "12"
os.environ["MASTER_API_CHAIN_STREAM_PROBE"] = "1.5"
os.environ["MASTER_API_CHAIN_COOLDOWN_BASE"] = "2"
os.environ["MASTER_API_CHAIN_COOLDOWN_MAX"] = "4"
os.environ["MASTER_API_CHAIN_FINAL_MIN"] = "3"

sys.path.insert(0, str(ROOT / "gateway"))

MOCK_PORT = 18100
GW_PORT = 18101
CLIENT_TOKEN = "sk-e2e-test-0001"

# ---------------------------------------------------------------- mock 上游
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

mock_app = FastAPI()

# 各候选最近一次实际收到的请求体（供测试断言"网关是否原样透传/剥离了参数"）
MOCK_SEEN: dict[str, dict] = {}


def _sse(chunks):
    async def gen():
        for c in chunks:
            yield f"data: {json.dumps(c, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@mock_app.post("/v1/chat/completions")
async def mock_chat(request: Request):
    body = await request.json()
    model = body.get("model", "")
    MOCK_SEEN[model] = body

    if model in ("flaky503a", "flaky503b"):
        return JSONResponse(status_code=503, content={
            "error": {"message": "分组 free 下无可用渠道（distributor）", "type": "upstream_error"}})

    if model in ("badparam", "badparam2"):
        # 参数不被接受：流式以 200+SSE 错误事件报出，非流式以 200+error body 报出
        if "reasoning" in body or "enable_thinking" in body:
            msg = "Validation: Unsupported parameter(s): `reasoning`, `enable_thinking`"
            if body.get("stream"):
                return _sse([{"error": {"message": msg}}])
            return JSONResponse(status_code=200, content={"error": {"message": msg}})
        if body.get("stream"):
            return _sse([
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "hello-e2e"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}},
            ])
        return JSONResponse(status_code=200, content={
            "id": "chatcmpl-mock", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello-e2e"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
        })

    if model == "toolonly":
        return _sse([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        ])

    if model == "hang":
        import asyncio as _aio
        await _aio.sleep(30)  # 200 已发出但迟迟不出内容 → 探测超时应切换
        return _sse([{"choices": [{"delta": {"content": "late"}}]}, ])

    # good / 其他
    if body.get("stream"):
        return _sse([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "hello-e2e"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}},
        ])
    return JSONResponse(status_code=200, content={
        "id": "chatcmpl-mock", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello-e2e"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    })


def _free_port(preferred):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


MOCK_PORT = _free_port(MOCK_PORT)
GW_PORT = _free_port(GW_PORT)


def _start_uvicorn(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    t0 = time.time()
    while not server.started and time.time() - t0 < 15:
        time.sleep(0.05)
    assert server.started, f"server on {port} failed to start"
    return server


# ---------------------------------------------------------------- 种子数据
def seed_db():
    import app.security as sec
    from app.database import execute, set_setting, fetch_one, get_setting

    ts = "2026-01-01T00:00:00.000+08:00"
    key_id = str(uuid.uuid4())
    ciphertext = sec.encrypt_upstream_key("mock-secret-key", get_setting("upstream_master_key"))
    execute(
        """INSERT INTO upstream_keys (id, name, provider, api_key_ciphertext, key_prefix, base_url,
           weight, rpm_limit, switch_threshold, status, created_at, updated_at)
           VALUES (?, 'mock', 'mock', ?, 'mock-', ?, 1, 0, 38, 'active', ?, ?)""",
        (key_id, ciphertext, f"http://127.0.0.1:{MOCK_PORT}/v1", ts, ts),
    )
    client_id = str(uuid.uuid4())
    execute("INSERT INTO clients (id, name, status, created_at, updated_at) VALUES (?, 'e2e', 'active', ?, ?)",
            (client_id, ts, ts))
    execute(
        """INSERT INTO client_api_keys (id, client_id, key_hash, key_prefix, key_ciphertext, status, created_at)
           VALUES (?, ?, ?, ?, 'x', 'active', ?)""",
        (str(uuid.uuid4()), client_id, sec.hash_secret(CLIENT_TOKEN),
         CLIENT_TOKEN[:13], ts),
    )
    chains = {
        "e2e-params": ["mock::flaky503a", "mock::badparam", "mock::good"],
        "e2e-params2": ["mock::badparam2", "mock::good"],
        "e2e-tool": ["mock::toolonly"],
        "e2e-hang": ["mock::hang", "mock::good"],
        "e2e-allfail": ["mock::flaky503a", "mock::flaky503b"],
    }
    models = [{"id": k, "display_name": k, "chain": v, "enabled": True} for k, v in chains.items()]
    set_setting("virtual_models", json.dumps(models, ensure_ascii=False))
    assert fetch_one("SELECT COUNT(*) c FROM upstream_keys")["c"] == 1


# ---------------------------------------------------------------- 测试主体
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  -- {detail}" if (detail and not cond) else ""))


def db_rows(sql, params=()):
    conn = sqlite3.connect(os.environ["MASTER_API_SQLITE_PATH"])
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


async def _amain():
    import httpx
    base = f"http://127.0.0.1:{GW_PORT}"
    headers = {"Authorization": f"Bearer {CLIENT_TOKEN}"}
    zcode_params = {"reasoning": {"effort": "max"}, "enable_thinking": True}
    async with httpx.AsyncClient(timeout=30) as client:

        async def chat(model, stream=True, with_params=True, virtual=None):
            body = {"model": virtual or model, "messages": [{"role": "user", "content": "hi"}],
                    "stream": stream}
            if with_params:
                body.update(zcode_params)
            r = await client.post(f"{base}/v1/chat/completions", headers=headers, json=body)
            return r

        def attempt_rows(model_ref):
            time.sleep(0.7)  # 尝试行由后台线程落库
            return db_rows("SELECT * FROM request_logs WHERE model=? AND log_category='chain_attempt'", (model_ref,))

        # ---- T1: 参数自适应（流式：200+SSE 流内 Validation 错误）----
        print("T1 参数自适应（流内 Validation → 剥离重试）")
        t0 = time.time()
        r = await chat("e2e-params", stream=True, virtual="e2e-params")
        text = r.text
        check("T1a 客户端拿到 200 与正常内容", r.status_code == 200 and "hello-e2e" in text, text[:200])
        check("T1b 错误未透传给客户端", "Unsupported parameter" not in text and "无可用渠道" not in text)
        check("T1c 整链耗时可接受(<10s)", time.time() - t0 < 10, f"{time.time()-t0:.1f}s")
        rows_badparam = attempt_rows("mock::badparam")
        check("T1d badparam 首次失败被记录(chain_attempt 502)",
              any(row["status_code"] == 502 for row in rows_badparam), str(rows_badparam)[:200])
        check("T1e 失败行请求体含原始参数",
              any("enable_thinking" in (row["request_body"] or "") for row in rows_badparam))
        # 思考强度保真：拒绝参数的候选被剥离（T3d 在 good 实际被调用后断言）
        check("T1f 拒绝候选收到的重试请求体已剥离",
              "enable_thinking" not in MOCK_SEEN.get("badparam", {}) and "reasoning" not in MOCK_SEEN.get("badparam", {}))
        n_before = len(attempt_rows("mock::badparam"))

        # ---- T1b: 黑名单跨请求 ----
        print("T1b 黑名单跨请求生效")
        r = await chat("e2e-params", stream=True, virtual="e2e-params")
        check("T1b-1 再次调用仍成功", r.status_code == 200 and "hello-e2e" in r.text)
        n_after = len(attempt_rows("mock::badparam"))
        check("T1b-2 badparam 不再产生失败行", n_after == n_before, f"{n_before} -> {n_after}")

        # ---- T2: 纯 tool_calls 流 ----
        print("T2 纯工具调用流不误判空流")
        r = await chat("e2e-tool", stream=True, virtual="e2e-tool")
        check("T2a 200 且工具调用完整透传",
              r.status_code == 200 and '"tool_calls"' in r.text and "[DONE]" in r.text, r.text[:200])
        check("T2b 无 upstream_empty_response", "upstream_empty_response" not in r.text)
        time.sleep(0.7)
        empty_rows = db_rows("SELECT * FROM request_logs WHERE model='e2e-tool' AND status_code>=400")
        check("T2c 无 502 空流日志", not empty_rows, str(empty_rows)[:200])

        # ---- T3: 挂起候选探测超时切换 ----
        print("T3 挂起候选探测超时快速切换")
        t0 = time.time()
        r = await chat("e2e-hang", stream=True, virtual="e2e-hang")
        dt = time.time() - t0
        check("T3a 200 且由 good 候选完成", r.status_code == 200 and "hello-e2e" in r.text)
        check("T3b 挂起候选未拖慢整链(<8s)", dt < 8, f"{dt:.1f}s")
        hang_rows = attempt_rows("mock::hang")
        check("T3c hang 尝试行状态 504", any(row["status_code"] == 504 for row in hang_rows), str(hang_rows)[:200])
        # good 在本请求中接管：它从未拒绝过参数，收到的请求体必须原样保留思考参数
        check("T3d 接受参数的候选参数原样透传（思考强度可生效）",
              "enable_thinking" in MOCK_SEEN.get("good", {}) and "reasoning" in MOCK_SEEN.get("good", {}),
              str(list(MOCK_SEEN.get("good", {}).keys())))

        # ---- T4: 全链失败状态码映射 ----
        print("T4 全链失败状态码映射")
        r = await chat("e2e-allfail", stream=True, virtual="e2e-allfail")
        try:
            detail = r.json()
            code = detail.get("error", {}).get("code", "")
        except Exception:
            code = ""
        check("T4a 返回 502（5xx 类别映射）", r.status_code == 502, str(r.status_code))
        check("T4b 业务码为 virtual_model_upstream_error", code == "virtual_model_upstream_error", code)

        # ---- T5: 非流式 200+error 体形式的参数错误（全新候选，未学习过）----
        print("T5 非流式参数错误剥离重试")
        r = await chat("e2e-params2", stream=False, virtual="e2e-params2")
        ok = r.status_code == 200 and "hello-e2e" in r.text
        check("T5a 非流式 200 且内容正确", ok, r.text[:200])
        rows = attempt_rows("mock::badparam2")
        check("T5b 首次失败行 502 且请求体含原始参数",
              any(row["status_code"] == 502 and "enable_thinking" in (row["request_body"] or "")
                  for row in rows), str(rows)[:300])
        rows = db_rows("SELECT * FROM request_logs WHERE model LIKE 'mock::%' ORDER BY started_at DESC LIMIT 12")
        check("T5c 候选尝试行 request_body 脱敏(messages省略)",
              all("OMITTED" in (row["request_body"] or "") or "messages" not in (row["request_body"] or "")
                  for row in rows if row["request_body"]))


def main():
    print(f"[e2e] 临时库: {TMPDIR}")
    from app.database import init_db, seed_defaults
    init_db()
    seed_defaults()
    mock = _start_uvicorn(mock_app, MOCK_PORT)
    seed_db()

    from app.main import app as gw_app
    gw = _start_uvicorn(gw_app, GW_PORT)
    print(f"[e2e] mock@{MOCK_PORT}  gateway@{GW_PORT}")

    import asyncio
    asyncio.run(_amain())

    print(f"\n===== E2E 结果: {len(PASS)} 通过 / {len(FAIL)} 失败 =====")
    if FAIL:
        print("失败项:", FAIL)
        print(f"[e2e] 临时库保留供排查: {TMPDIR}")
        sys.exit(1)
    # 成功才清理临时库
    try:
        (TMPDIR / "e2e.db").unlink()
        TMPDIR.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
