# Master API — 本地多供应商 AI 网关架构文档

> **最后更新**: 2026-10-06
> **语言**: 中文（简体）

---

## 1. 项目定位

Master API 是一个**本机自用的多供应商 AI 网关**，运行在 `127.0.0.1:8000`（HTTP）。

它解决的问题很具体：手上有多个不同厂商/中转的 API Key（各自有不同的 Base URL、协议格式、限流策略），希望把它们揉成一个统一的 OpenAI 兼容入口给本机的各种 AI 客户端使用，并且：

- **聚合模型**：把多个真实模型串成一条故障转移链，对外暴露一个虚拟模型 ID，链上模型不可用时自动切换下一个；
- **密钥安全**：上游 Key 加密存储，界面不回显明文；
- **可观测**：每次请求落日志，控制台可视化用量与状态。

---

## 2. 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 运行时 | Python 3.14 + venv | 单机部署，无需 Docker |
| Web 框架 | FastAPI + uvicorn | 全异步，本地 HTTP 服务 |
| 数据库 | SQLite（WAL 模式） | 单文件 `master.db` |
| 加密 | cryptography（Fernet + HKDF-SHA256） | 上游/客户端密钥加密 |
| 认证 | bcrypt（管理员密码）+ HMAC Token（24h） | 控制台登录 |
| HTTP 客户端 | httpx + tenacity | 异步转发上游、指数退避重试 |
| 前端 | 原生 HTML/CSS/JS（单文件控制台） | 无构建步骤 |

---

## 3. 目录结构

```
master-api/
├── gateway\
│   └── app\
│       ├── main.py             # FastAPI 入口：/、/admin、/admin/console、/healthz
│       ├── public_api.py       # /v1/* 公共 API（模型列表、对话、embeddings）
│       ├── admin_api.py        # /gw/admin/* 管理 API（约 40+ 端点）
│       ├── admin_panel.py      # /gw/dbadmin SQLAdmin 数据库面板
│       ├── scheduler.py        # SurgeScheduler：调度核心（冷却/熔断/权重）
│       ├── virtual_models.py   # 聚合模型（故障转移链）
│       ├── provider_probe.py   # 上游连通性探测（四种 API 格式，含 Responses）
│       ├── provider_presets.py # 内置供应商接入目录（含 ChatGPT 账号登录预设）
│       ├── openai_oauth.py     # Sign in with ChatGPT 登录流 + token 刷新
│       ├── responses_bridge.py # chat/completions ↔ OpenAI Responses API 协议桥接
│       ├── security.py         # Fernet+HKDF 加密、管理员认证
│       ├── database*.py        # 数据库访问层
│       ├── static\console.html # 管理控制台（单文件前端）
│       └── static\brand.png    # 品牌头像/favicon
├── master.db                # SQLite 数据库（全部业务数据）
├── .env                     # 配置（见第 8 节）
├── restart.bat              # 重启（杀 8000 端口旧进程 → 启动，拒绝访问时才提权）
├── start_local.bat          # 启动（普通权限）
├── stop.bat                 # 停止（杀 8000 端口进程，日志 %TEMP%\master-api-stop.log）
```

---

## 4. 请求流转

```
客户端（任意 OpenAI 兼容工具）
        │  POST http://127.0.0.1:8000/v1/chat/completions
        │  Authorization: Bearer <客户端 Key>
        ▼
┌─────────────────────────────────────────────┐
│ 1. 客户端认证                                 │
│    client_api_keys 表校验 → 记录 last_used   │
├─────────────────────────────────────────────┤
│ 2. 模型解析                                   │
│    普通模型 → 直接路由到声明它的供应商          │
│    虚拟模型 → 展开聚合链（按顺序故障转移）       │
├─────────────────────────────────────────────┤
│ 3. 供应商调度（SurgeScheduler，公平调度）      │
│    健康过滤：冷却中 / 隔离中 / RPM 超限的      │
│    组合不参与候选，其余按公平排序选出           │
│    provider::model 组合                       │
├─────────────────────────────────────────────┤
│ 4. 上游转发                                   │
│    httpx 异步 + tenacity 退避重试              │
│    流式(SSE)/非流式透传                        │
├─────────────────────────────────────────────┤
│ 5. 失败处理                                   │
│    请求级错误 → 链内切换下一个模型               │
│    429 → 桶级冷却（5s 快速恢复）                │
│    30s 内 429≥20 次 → 模型级熔断 10s           │
│    连续 3 次连接失败 → 隔离 5 分钟               │
├─────────────────────────────────────────────┤
│ 6. 落日志                                     │
│    request_logs：状态码/延迟/token/模型/客户端  │
└─────────────────────────────────────────────┘
```

对外端点（均为 OpenAI 兼容格式）：

| 端点 | 说明 |
|------|------|
| `GET /v1/models` | 模型列表（真实模型 + 聚合模型） |
| `POST /v1/chat/completions` | 对话补全（流式/非流式） |
| `POST /v1/embeddings` | 向量接口 |
| `GET /healthz` | 健康检查 |

---

## 5. 供应商与密钥管理

### 5.1 数据模型（upstream_keys 表）

| 字段 | 说明 |
|------|------|
| name / provider | 显示名 / 路由标识（唯一） |
| base_url | 自定义上游地址（任意 OpenAI/Anthropic/Gemini 兼容服务） |
| api_key_ciphertext | Fernet+HKDF 加密后的 Key（库内永不明文） |
| key_prefix | 脱敏前缀（如 `sk-G***D7O0`），用于界面展示 |
| weight / rpm_limit | 调度参数（权重、每分钟请求上限） |
| last_health_check_* | 最近连通性测试结果 |

### 5.2 连通性测试（provider_probe）

- 四种 API 格式自适应：OpenAI（Bearer）/ OpenAI Responses（Bearer + /responses 桥接）/ Anthropic（x-api-key）/ Gemini（x-goog-api-key）；
- 无模型时执行 `GET /models` 拉取上游模型列表；指定模型时发送最小对话请求实测；
- 内置 Key 清洗（剔除粘贴误带的空白与 `Bearer ` 前缀）、空白字符检测、非库 UA（规避上游 WAF）。

### 5.3 Key 生命周期

创建/编辑时自动清洗入库；支持"编辑供应商时替换 Key"（留空不改）；测试状态随配置变更自动失效重测；保存后自动补一次连通性检测。

### 5.4 内置接入目录（provider_presets）

- 静态目录 `provider_presets.py` 内置 26 家上游定义（主流大厂 / 国内平台 / 海外平台 / 聚合与推理云），每条含 id / name / baseUrl / Key 环境变量 / API 类型；
- 控制台"添加模型供应商"弹窗顶部新增**接入方法**下拉：选中一家自动预填名称、Base URL、API 格式与 Provider 标识（标识被占用时自动加 `-2` 后缀），全部字段仍可手改；不选则走原有手动创建流程（原有流程完全保留）；
- 预设的 Base URL 按网关转发契约（统一 OpenAI 协议 `{base_url}/chat/completions` + Bearer）对准可跑通端点：走原生 Anthropic/Conversations 协议的供应商填其 OpenAI 兼容路径（如 Gemini 用 `/v1beta/openai`）；
- 管理 API：`GET /gw/admin/upstreams/presets`（只读静态数据）。

### 5.5 ChatGPT 账号登录（OpenAI OAuth，openai-codex）

- **接入方式**：预设「OpenAI (ChatGPT 账号登录)」→ 浏览器完成 Sign in with ChatGPT 授权（动态客户端 `dynamic_agent_client` + PKCE S256，本地回调 `http://127.0.0.1:1455/auth/callback`；端口被占用时可手动粘贴回调 URL 兜底）。无需 API Key，凭据为 ChatGPT 订阅（Plus/Pro）的用户 access/refresh token；
- **数据模型**：`upstream_keys.auth_type`（`api_key` | `oauth`，自动迁移）；OAuth 凭据以加密 JSON 信封存入 `api_key_ciphertext`（含 access/refresh/expires_at/client_id，过期前 3 分钟刷新），`reveal` 只回概要且 refresh_token 打码；
- **协议桥接**（`responses_bridge.py`）：该凭据只能调 api.openai.com 的 **Responses API**（订阅计费），转发前把 chat/completions 请求翻成 `/responses`（system→developer、tools 平铺、max_output_tokens 下限 16、丢弃 temperature/top_p、reasoning_effort→reasoning.effort、store:false），响应（含 SSE 流式）翻回 chat.completion[.chunk]，usage/reasoning token 一并映射；工具调用（function calling）双向支持；
- **token 刷新**（`openai_oauth.py`）：转发/探测前 `ensure_fresh_access()` 自动续期并回写数据库、失效调度器密钥缓存；
- **管理 API**：`POST /gw/admin/upstreams/oauth/start`（发起登录）、`GET /upstreams/oauth/status/{state_id}`（轮询）、`POST /upstreams/oauth/paste`（手动粘贴回调 URL）、创建走 `POST /upstreams`（`auth_type:"oauth"` + `oauth_state_id`）；
- **代理**：访问 OpenAI（登录/刷新/转发/探测）可走代理——控制台侧边栏「代理设置」页、`admin_settings.oauth_proxy` 或 `.env` 的 `MASTER_API_OAUTH_PROXY`（如 `http://127.0.0.1:7890`）；restart.bat 会清空系统代理环境变量，直连被 OpenAI 地区封锁时必须配置；socks5 支持依赖 `socksio` 包（已装入 venv）；
- 默认预填订阅模型 gpt-5.1-codex / gpt-5.1 / gpt-5.1-codex-max / gpt-5.1-codex-mini；不支持 embeddings。

---

## 6. 聚合模型（虚拟模型）

- 存储：`admin_settings.virtual_models`（JSON），每条含 `id / display_name / chain / enabled`；
- 链元素格式 `provider::model`；
- **多模式调度路由策略（支持自定义配置）**：
  - **`round-robin` 平滑动态轮询（默认推荐）**：每个请求依次轮换起始尝试的模型，使流量在链内健康模型间平摊分发，彻底杜绝流量死锁在首个模型；若当前选中的模型故障，自动顺延尝试链内剩余健康模型。
  - **`priority` 主备优先容灾**：严格按配置先后顺序尝试，优先使用首选主模型，遇故障才向后降级。
  - **`adaptive` 自适应感知**：结合各候选的熔断状态、连续失败次数与延迟动态分发。
- **错误分级切换**（速度与容错平衡）：
  - 超时/挂起类错误对同一候选只试 1 次就切下一个（再等一轮完整超时只会成倍放大延迟）；429/401/403/5xx/网络抖动等瞬时错误保留 2 次换 key 重试，容忍偶发故障；
  - no_key（该候选暂无可用密钥）直接换下一个候选，且**不计入**模型失败——密钥冷却 ≠ 模型坏了，避免误伤健康模型；
- **候选超时预算**：链中非末位候选的等待时间封顶（`MASTER_API_CHAIN_CANDIDATE_TIMEOUT`，默认 60s；流式为"等到响应头"预算），末位兜底候选受整链剩余预算约束（保底 `MASTER_API_CHAIN_FINAL_MIN`=30s）——前面快速试错、兜底也有一次像样的尝试，但整链不会无限拖长；
- **整链预算**：链开始后超过 `MASTER_API_CHAIN_TOTAL_BUDGET`（默认 90s，对齐客户端/中间代理的超时窗口），跳过剩余"还有备选"的候选；整链最坏耗时 ≈ 预算 + 30s，保证客户端先收到网关的确定结果而不是自己代理的 504/524；
- **自适应软熔断**：60s 窗口内失败 <3 次（偶发）不触发任何惩罚；达阈值后冷却按连续触发次数指数升级（`MASTER_API_CHAIN_COOLDOWN_BASE`=30s → 60 → 120 → … 封顶 300s）；冷却到期进入**试探期**，放行但试探失败立即再次熔断（不重新累计窗口），成功一次完全恢复；
- **流式提交前探测**：聚合上游常见"先回 HTTP 200、再在 SSE 流里吐 error 事件（无可用渠道）"、"只发标准角色块后挂起"、"200 后迟迟不出首块"——一旦把 200 发给客户端就无法再切换。因此链内流式候选在提交前先缓冲并检查流首段（`MASTER_API_CHAIN_STREAM_PROBE`，默认 30s）：**必须等到真实内容块（文本/推理/工具调用）才提交**，角色块不算数；缓冲行先重放给客户端，发现 error/空流/超时未出内容即静默关闭并切换下一个候选；流已提交后的中途错误仍不可切换，但失败行会以 502/504 落库且附完整尝试链 JSON（控制台渲染时间线），不再无影无踪；
- **候选参数自适应**：客户端（如 ZCode）常附带 `reasoning`/`enable_thinking`/`reasoning_effort` 等非标准参数，各上游接受度不一，且拒绝可能以 HTTP 400 **或 200+SSE 流内错误**两种形式出现。网关在所有错误面（HTTP 400、流式探测、流中途错误、非流式 200+error 体）识别 "Unsupported parameter(s)" 类消息：解析出参数名 → 从请求体剥离 → 同候选原地重试（不计模型失败），并记入该候选的黑名单（内存，跨请求生效，后续请求预防性剥离）。**剥离严格按候选生效**：接受参数的候选（思考强度可正常生效的模型）永远原样透传，不受其他候选的拒绝影响。`model/messages/stream/tools` 等关键字段永不剥离；
- **提交后中途失败**：流式一旦提交（探测到真实内容块），上游再死亡物理上无法切换模型——但失败会以 502/504 落库（含完整尝试链 JSON），同时补记模型级软熔断失败，反复"晚死"的候选会被统计捕获、降低再次被选中的概率；
- **失败可观测**：链内每个候选尝试都会独立写入一行请求日志（`log_category='chain_attempt'`，模型列为 `provider::model` 真实引用，状态码为该次上游真实状态/超时映射），聚合失败时在"请求日志"里能直接看到每个备用模型的报错；最终返回行（虚拟模型 ID）的错误信息里附完整尝试链 JSON，控制台详情弹窗渲染成时间线表格；统计接口（仪表盘/摘要/趋势/Top 榜）统一排除 chain_attempt 行，避免请求数与延迟被链内多次尝试稀释；
- **状态码保真**：全链失败时按最后一个候选的失败类别映射返回状态码——超时→504、限流→429、上游 5xx→502、其余→503，请求日志与客户端实际收到的一致；
- 控制台列表只展示"当前使用模型"（链中第一个未熔断的）+ 备用数量，编辑弹窗内可完整调整链顺序。

---

## 7. 调度机制（SurgeScheduler 实际在用的全部逻辑）

调度核心 `select_key(model)` 采用**公平调度**：健康性评分只用于过滤，最终在健康候选中公平排序选择。

| 机制 | 触发条件 | 行为 | 恢复 |
|------|----------|------|------|
| **429 桶级冷却** | 上游返回 429 / 403 / 超时 | 该 (key, model) 组合进入冷却，期间不参与候选（429→5s；403→60s 起自适应退避，上限 600s；超时→15s） | 冷却到期自动恢复 |
| **模型级 429 熔断** | 30 秒窗口内同一模型 429 ≥ 20 次 | 该模型直接向客户端返回 429，避免反复尝试的恶性循环 | 10 秒后自动解除 |
| **连接失败隔离** | 连续 3 次连接失败 / SSL 错误 | 该 (key, model) 隔离 5 分钟（权重归零） | 隔离到期自动恢复 |
| **RPM 限制** | 供应商配置的 rpm_limit | 超限的组合跳过本轮调度 | 下一窗口自动恢复 |
| **链内切换** | 超时/模型不存在/无可用密钥（即时切）；429/5xx 等瞬时错误（同候选换 key 重试 2 次后切） | 聚合模型展开时切换到链中下一个模型，非末位候选等待封顶 60s | 即时 |

辅助机制（仅观测，不拦截）：客户端并发计数（`record/release_client_request`）供控制台展示；连接池复用（`get_http_pool` / `get_stream_pool`）。

---

## 8. 管理控制台

### 8.1 页面（http://127.0.0.1:8000/admin/console）

| 页面 | 功能 |
|------|------|
| 网页对话 | 独立网页聊天板块：会话侧栏（今天/7天内/30天内/更早分组）、开启新对话、重命名/删除、图片+文档输入、流式输出、思考内容折叠框、蓝色链接+网页图标、发送键旁模型/推理等级选择器（直连全部上游模型，免 API Key） |
| 仪表盘 | 总/月/周/日 Token、模型用量分布、成功率、平均延迟 |
| 上游密钥 | 供应商 CRUD、接入方法预设（内置目录一键预填）、模型拉取（在线/手动）、连通性测试、启停 |
| 聚合模型 | 聚合链 CRUD、链测试、当前使用模型展示 |
| 创建 API Key | 下游客户端 Key 的签发/启停/明文查看 |
| 请求日志 | 全量请求明细、详情弹窗、统计摘要、清理 |
| 代理设置 | 网关出站代理配置（直连/http/socks5）、出口 IP 连通性测试、即时生效 |

> 网页对话数据存于 `chat_sessions` / `chat_messages` 表；流式开关存于 `admin_settings.chat_config`。
> 浏览器直接调本网关 `/v1/chat/completions`，Bearer = 管理员登录 Token（`authenticate_client` 对管理员 Token 直接放行，网页对话无需配置网关 API Key）；外部客户端仍走 `acu_`/`sk-` Key 鉴权。

### 8.2 管理 API（/gw/admin/*）

认证：`POST /gw/admin/login`（密码 = `.env` 中 `MASTER_ADMIN_PASSWORD_HASH` 对应明文）→ 签发 24h Token（HMAC-SHA256），后续请求 `Authorization: Bearer <token>`。

主要端点组：`/upstreams`（CRUD+接入目录 presets+ChatGPT 账号登录 oauth/*+模型+测试+健康检查）、`/proxy`（出站代理读取/保存/测试）、`/virtual-models`（CRUD+测试）、`/clients` 与 `/clients/{id}/keys`（下游客户端）、`/request-logs`（查询/详情/统计/清理）、`/audit-logs`、`/global-status`、`/circuit-breakers`、`/chat/*`（网页对话：sessions CRUD、messages 读写、truncate 截断、config 读写）。

### 8.3 网页对话

- 首条用户消息自动生成会话标题（截取前 30 字），也可手动重命名；
- 消息附件（图片 dataURL / 文档文本 / PDF base64）随消息存入 `chat_messages.attachments`（JSON，单条上限 8MB）；图片前端先压缩（最长边 2048px，JPEG 85%）；
- 发给 `/v1/chat/completions` 时：图片转 `image_url` content part、文本类文档内联为文本、PDF 转 `file` content part（OpenAI 格式）；历史消息中的图片/PDF 只保留最近一条用户消息的，其余替换为占位文本，避免请求体无限膨胀；
- 数学公式：内置 KaTeX 0.16.9（`static/katex/`，本地离线可用）；Markdown 渲染前先提取 `$$...$$`、`\[..\]`（独立公式）与 `$...$`、`\(...\)`（行内公式）为占位符再走块级解析，行内代码里的 `$` 不当公式；代码块/思考框内保持源码原样，无定界符的裸 LaTeX 不渲染；
- 流式：浏览器 fetch + ReadableStream 解析 SSE，`delta.reasoning_content` 渲染进思考框（思考中展开、完成后自动收纳并可点击回看），`delta.content` 实时渲染 Markdown；支持中途停止（AbortController，已生成部分照常入库）；
- 生成任务按会话隔离（`CS.tasks`）：生成中可随时开新对话/切会话，任务转后台继续并回写原会话；发送后至首个输出到达前显示"调用中…………"流动光效提示；
- 用户气泡下有 复制 / 铅笔（rewind 编辑重发）操作：编辑后先调 `POST /chat/sessions/{id}/truncate`（保留前 keep 条、删除其余，含审计）截断旧分支，再以新文案走正常发送链路；编辑首条消息且标题为自动派生时同步会话标题；
- 模型选择：发送键旁选择器直接列出「上游密钥」里全部供应商的真实模型（`GET /available-models`，按供应商分组，`provider::model` 精确锁定上游），选择记忆存浏览器 localStorage，不受聚合模型限制；推理等级（reasoning_effort）可选，default 不传参；
- 首页 `http://127.0.0.1:8000/` 只显示"配置网关请到管理员后台"提示页，不再提供试玩对话。

### 8.4 SQLAdmin

`/gw/dbadmin`：数据库全表 CRUD（同一管理员密码认证），用于直接修数据。

---

## 9. 配置（.env）

| 变量 | 说明 |
|------|------|
| `DATABASE_BACKEND=sqlite` | 存储后端（本机固定 sqlite） |
| `MASTER_API_SQLITE_PATH=master.db` | 数据库文件 |
| `MASTER_ADMIN_PASSWORD_HASH` | 管理员密码 bcrypt 哈希（12 rounds） |
| `MASTER_API_LOCAL_DEMO_KEY` | 本地 Chat 测试页预填的客户端 Key |
| `GW_BASE_URL=http://127.0.0.1:8000` | 网关自身地址 |
| `MASTER_API_OAUTH_PROXY` | 可选：访问 OpenAI（ChatGPT 账号登录/转发）的代理，如 `http://127.0.0.1:7890`；也可在 admin_settings 设 `oauth_proxy` |
| `CORS_ALLOWED_ORIGINS` | CORS 白名单（含 http/https 本机来源） |
| `ADMIN_SESSION_SECRET` | 数据库面板（/gw/dbadmin）会话签名密钥 |

> ⚠️ 修改密码哈希或密钥类配置后需重启网关生效。

---

## 10. 安全设计

- **上游/客户端密钥**：Fernet（HKDF-SHA256 派生）加密存储；界面只显示脱敏前缀，明文需显式"查看"并记审计日志；
- **管理员认证**：bcrypt 哈希比对，不落明文；Token 有效期 24h；
- **传输**：纯本地 HTTP（绑定 127.0.0.1 回环地址，不出网，第三方客户端开箱即用，无需配置自签名证书信任）；
- **网络面**：仅绑定 127.0.0.1，外网不可达；CORS 白名单；
- **审计**：管理操作（创建/删除/查看明文等）全部写入 audit_logs。

---

## 11. 部署与运维

### 11.1 启动/停止

| 方式 | 说明 |
|------|------|
| 桌面「Master API 重启」 | `restart.bat`：杀 8000 端口旧进程 → 加载 .env → 启动（前台窗口，关窗即停）；旧进程拒绝访问时才请求提权 |
| 桌面「停止 Master API」 | `stop.bat`：杀 8000 端口进程（无进程则提示），日志 `%TEMP%\master-api-stop.log` |
| 手动 | `start_local.bat`（免提权直接启动） |

### 11.2 健康检查与排障

```bash
curl http://127.0.0.1:8000/healthz          # 存活
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer <客户端Key>"
```

- 请求级排障：控制台 → 请求日志 → 详情；
- 连通性排障：上游密钥 → 测试连通性（错误信息含上游原始响应）；
- 停止脚本排障：`%TEMP%\master-api-stop.log`。

### 11.3 备份

业务数据全部在 `master.db`（含 -wal/-shm），停止网关后拷贝该文件即可完整备份。

---

> **文档版本**: 2026-10-04 重写，与代码实态对齐
