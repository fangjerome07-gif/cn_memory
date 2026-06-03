# cn_memory

Hermes Agent 的持久化、零依赖中文记忆插件。通过 LLM 自动提取 + 混合检索，让 AI 拥有跨会话的长期记忆。

## 为什么选择 cn_memory？

大多数 AI 助手在会话之间会遗忘一切。`cn_memory` 用两个自动循环解决这个问题：

1. **每次对话自动提取记忆** — 对话结束时，插件自动用 LLM 分析整段对话上下文，抽取高价值事实（偏好、规则、项目、健康、工具等 8 大类），存入本地 SQLite 数据库。无需手动保存。
2. **每次对话自动注入相关记忆** — 对话开始时，插件自动检索最相关的记忆，注入系统提示词。你的 Agent 从此拥有跨会话的持久记忆。

这两个循环在后台静默运行。Agent 只是……记住了。

## 核心特性

### 架构优势

- **零基础设施依赖** — 只需 **SQLite** + **numpy**。不需要 Docker、PostgreSQL、Qdrant、Neo4j。`pip install` 即可运行。
- **中文（CJK）深度优化** — 针对中文文本设计，使用 **Trigram 分词** 进行 FTS5 全文检索，配合 **LIKE 模糊查询** 兜底，最大化中文召回率。
- **多 Agent 隔离** — 按 Agent 身份自动创建独立 SQLite 数据库，互不干扰。

### 检索与打分

- **混合检索** — 结合稠密向量搜索（numpy 余弦相似度）+ 稀疏关键词匹配（SQLite FTS5 BM25）+ 模糊 LIKE 搜索。
- **六维打分公式**：
  - 向量相似度：**40%**
  - BM25 关键词匹配：**30%**
  - LIKE 模糊匹配：**20%**
  - 基础重要性：**15%**
  - 时间衰减（近因效应）：**15%**
  - 类型权重：**10%**
- **规则最高优先级** — 标记为 `rule` 类型的记忆权重为 **1.0**，确保关键规则永远不会被忽略。

### 记忆生命周期

- **自动去重与覆盖**：
  - 相似度 > 0.98：自动合并（消除重复）
  - 相似度 0.92–0.98：旧记忆标记为 `superseded`，被更新更准确的条目取代
- **定期画像汇总** — 每 50 次写入，自动用 LLM 将零散事实聚合为结构化用户画像。
- **TTL 过期与自动归档** — 支持为每条记忆设置存活时间。过期记忆自动归档（不删除），保留历史记录。
- **8 大记忆类别**：`preference`（偏好）、`profile`（画像）、`project`（项目）、`rule`（规则）、`health`（健康）、`tool`（工具）、`relationship`（关系）、`general`（通用）

### 附加功能

- **内置 Todo 追踪** — 自动从对话中提取待办事项，支持状态机管理（`open` → `done` → `cancelled`）。

## 安装

1. 将本插件复制到 Hermes 插件目录：
   ```bash
   cp -r cn_memory ~/.hermes/plugins/
   ```
2. 安装依赖：
   ```bash
   pip install numpy pyyaml
   ```
   *（SQLite 和其他标准库已包含在 Python 中，无需额外安装。）*

## 配置

### Hermes config.yaml

```yaml
memory:
  provider: cn_memory
  memory_char_limit: 2200  # 注入记忆的最大字符数
```

### 插件配置文件

创建 `~/.hermes/cn_memory/config.json`：

```json
{
  "embedding_endpoint": "http://127.0.0.1:18080/v1/embeddings",
  "embedding_model": "bge-small-zh-v1.5",
  "memory_char_limit": 2200,
  "llm_provider": "openai",
  "llm_base_url": "https://api.openai.com/v1",
  "llm_model": "gpt-4o-mini",
  "llm_api_key": "your-api-key-here",
  "llm_timeout": 8.0
}
```

### 环境变量

| 变量名 | 说明 | 默认值 |
|---|---|---|
| `CN_MEMORY_EMBEDDING_ENDPOINT` | 嵌入向量 API 端点（兼容 OpenAI） | `http://127.0.0.1:18080/v1/embeddings` |
| `CN_MEMORY_EMBEDDING_MODEL` | 嵌入模型名称 | `bge-small-zh-v1.5` |
| `CN_MEMORY_CHAR_LIMIT` | 注入记忆的最大字符数 | `2200` |
| `CN_MEMORY_LLM_BASE_URL` | LLM API 基础地址（用于事实抽取） | 无 |
| `CN_MEMORY_LLM_MODEL` | LLM 模型名称（用于事实抽取） | 无 |
| `CN_MEMORY_LLM_API_KEY` | LLM API 密钥（用于事实抽取） | 无 |
| `CN_MEMORY_LLM_TIMEOUT` | LLM 请求超时时间（秒） | `8.0` |

## 使用

### 工具（自动注册）

**`cn_memory_store`** — 存储结构化事实：
- `content`（字符串，必填）：要记住的内容
- `memory_type`（字符串）：`preference`、`profile`、`project`、`rule`、`health`、`tool`、`relationship`、`general`
- `importance`（整数）：1（低）到 5（关键）
- `confidence`（浮点数）：0.0 到 1.0
- `ttl_days`（浮点数）：过期天数（可选）

**`cn_memory_search`** — 搜索记忆：
- `query`（字符串，必填）：搜索关键词或语义描述

### CLI 查看工具

```bash
python cn_memory_viewer.py --profile default
```

## 工作原理

```
对话结束
    │
    ▼
┌─────────────────────┐
│ LLM 事实抽取         │  ← 分析整段对话
│ （8 大类别）          │  ← 提取结构化事实
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ SQLite 存储          │  ← 向量嵌入 + FTS5 索引
│ （按 profile 分库）   │  ← 自动去重与覆盖
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 画像定期汇总         │  ← 每 50 次写入触发
│ （LLM 聚合）         │  ← 生成结构化用户画像
└─────────────────────┘


对话开始
    │
    ▼
┌─────────────────────┐
│ 混合检索             │  ← 向量 + BM25 + LIKE
│ （query → 记忆）     │  ← 六维打分排序
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 注入系统提示词       │  ← Top-K 记忆注入 prompt
│ （2200 字上限）       │  ← Agent "记住"上下文
└─────────────────────┘
```

## 竞品对比

| 特性 | cn_memory | Mem0 | Hindsight |
|---|---|---|---|
| 基础设施 | **仅需 SQLite** | Qdrant + Neo4j | PostgreSQL + pgvector |
| 中文（CJK）优化 | **Trigram + LIKE 兜底** | ❌ | ❌ |
| 每次对话自动提取 | **✅ LLM 驱动** | ✅ | ✅ |
| 每次对话自动注入 | **✅ 注入系统提示词** | ✅ | ✅ |
| 打分透明度 | **可配置权重** | 黑盒 | 反思机制 |
| Todo 追踪 | **✅ 内置** | ❌ | ❌ |
| 部署方式 | **pip install** | 推荐 Docker | 必须 Docker |

## 许可证

[MIT 许可证](LICENSE)
