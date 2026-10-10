# Master API — 本地多供应商 AI 网关

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)

> **Master API** 是一个专为个人本地开发打造的**多供应商 AI 聚合网关**。将手头多个不同厂商 / 中转的 API Key 统一到一个 OpenAI 兼容接口，支持**聚合模型故障转移链**、密钥本地加密存储、智能熔断与请求全量日志追踪。

---

## 核心特性

- 🔄 **多供应商接入**：支持任意 OpenAI / Anthropic / Gemini 兼容端点，自由配置 Base URL 与协议格式；
- 🔗 **聚合模型与故障转移**：按 `provider::model` 组装故障转移链，主模型异常或超额自动无缝重试下一个；
- ⚡ **智能调度与熔断**：支持权重调度、RPM 限制、429 自动冷却退避、连续失败自动隔离；
- 🔍 **一键连通性测试**：控制台即点即测供应商与模型健康度，精准透传上游原始错误；
- 🔐 **高强度密钥加密**：基于 Fernet + HKDF-SHA256 加密入库，前端严格脱敏展示，查看明文触发审计日志；
- 📊 **全量请求观测**：记录 HTTP 状态码、响应耗时、Token 用量、调用模型及客户端标识，全链路可审计；
- 🎨 **原生现代化控制台**：内置轻量控制台（Mono Premium 黑白双主题），单文件前端，零构建开箱即用；
- 🏠 **纯本地安全运行**：默认绑定 `127.0.0.1:8000`，无需繁杂证书配置，各类第三方客户端直接连接。

---

## 快速安装与启动

### 1. 环境准备
确保本机已安装 **Python 3.10+** 与 **Git**。

### 2. 克隆项目并安装依赖
```bash
# 克隆仓库
git clone https://github.com/<your-username>/master-api.git
cd master-api

# 创建并激活虚拟环境 (可选但推荐)
# Windows:
python -m venv venv
venv\Scripts\activate

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 初始化配置文件
复制配置模板生成 `.env`：
```bash
# Windows:
copy .env.example .env

# Linux / macOS:
cp .env.example .env
```
> 默认管理控制台初始登录密码为：`admin12345`。如需修改，可参考 `.env.example` 中的注释说明。

### 4. 启动服务

- **Windows 用户**：可以直接双击运行根目录的 `start_local.bat`。
- **跨平台 / 命令行启动**：
  ```bash
  uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir gateway
  ```

启动成功后：
- 🎛️ **管理控制台**：<http://127.0.0.1:8000/admin/console>
- 🔌 **OpenAI 兼容端点**：<http://127.0.0.1:8000/v1>

---

## 快速使用流程

1. **登录控制台**：浏览器访问 `http://127.0.0.1:8000/admin/console`，输入管理员密码（默认 `admin12345`）；
2. **添加上游密钥**：在「上游密钥」页面添加供应商（填入 Base URL 与 API Key，可一键拉取或手动输入模型列表）；
3. **配置聚合模型**：在「聚合模型」页面设置下游调用的虚拟模型别名，并串联故障转移序列；
4. **签发客户端 Key**：在「创建 API Key」为你的下游工具（如 Claude Dev, Cline, NextChat, LibreChat 等）生成密钥；
5. **客户端接入**：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer <你的客户端Key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-aggregate-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 常用环境配置（.env）

| 配置项 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `DATABASE_BACKEND` | 数据库类型（`sqlite` 或 `postgres`） | `sqlite` |
| `MASTER_API_SQLITE_PATH` | SQLite 数据文件存储路径 | `master.db` |
| `MASTER_ADMIN_PASSWORD_HASH` | 管理员密码 bcrypt 12 rounds 哈希 | 对应 `admin12345` |
| `MASTER_API_LOCAL_DEMO_KEY` | 本地自测用 Demo API Key | `sk-...` |
| `ADMIN_SESSION_SECRET` | 控制台 Session 鉴权密钥 | 随机字符串 |
| `GW_BASE_URL` | 网关基础地址 | `http://127.0.0.1:8000` |
| `CORS_ALLOWED_ORIGINS` | CORS 允许来源 | 本地常见端口 |

详细架构设计与实现见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 项目目录结构

```
master-api/
├── gateway/
│   ├── app/                # FastAPI 核心业务逻辑
│   │   ├── admin_api.py    # 管理后台 API
│   │   ├── public_api.py   # OpenAI 兼容代理网关接口
│   │   ├── scheduler.py    # 模型分发与故障转移调度引擎
│   │   ├── security.py     # 密钥加密与鉴权
│   │   ├── database_sqlite.py # SQLite 数据持久化与自建表
│   │   └── static/         # 前端控制台静态资源
│   └── requirements.txt
├── .env.example            # 配置项模板
├── .gitignore              # Git 忽略规则（保护数据库与密钥）
├── requirements.txt        # Python 依赖清单
├── restart.bat             # Windows 便捷重启脚本
├── start_local.bat         # Windows 启动脚本
├── stop.bat                # Windows 停止脚本
├── ARCHITECTURE.md         # 架构设计文档
└── LICENSE                 # 开源协议 (MIT)
```

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源发布。
