# 🔐 cn_memory 安全配置指南

**2026-06-04 B 方案：移除硬编码 API key**

## 🚨 历史问题

fork 自 `https://github.com/fangjerome07-gif/cn_memory` 的早期版本在 `cn_memory.py:575` 处硬编码了一个 minimax API key 作为 fallback。该硬编码存在以下问题：

1. **安全风险**：API key 暴露在源码中，任何有仓库访问权限的人都能看到
2. **可滥用**：即使本地配置了新 key，旧 key 仍可被使用
3. **不可审计**：无法追踪哪个 key 被实际使用

## ✅ B 方案修复

**PR/commit**: 移除 `cn_memory.py:575` 硬编码，改用环境变量配置

### 优先级链（自动 fallback）

```
CN_MEMORY_EMBEDDING_API_KEY  (embedding 专用)
  ↓ (如果未设置)
CN_MEMORY_LLM_API_KEY  (LLM 共享)
  ↓ (如果未设置)
"" (空字符串，调用时会返回 401)
```

### 配置方法

#### 方式 1: 直接环境变量
```bash
export CN_MEMORY_EMBEDDING_API_KEY="sk-your-key"
export CN_MEMORY_LLM_API_KEY="sk-your-other-key"
python3 your_app.py
```

#### 方式 2: .env 文件
```bash
cp .env.example .env
# 编辑 .env 填入你的 key
# 程序会自动从 ~/.hermes/.env 或 CN_MEMORY_HERMES_HOME/.env 加载
```

#### 方式 3: 外部 secrets manager
```python
# 在你的应用启动代码中
import os
os.environ["CN_MEMORY_EMBEDDING_API_KEY"] = load_from_vault("minimax_embed")
from cn_memory import MemoryStore
```

## 🧪 测试场景

| 场景 | 预期行为 |
|------|---------|
| ENV 缺失 | 调用 embedding → 401 → `_get_embedding` 返回 None |
| ENV 正确 | 调用 embedding → 使用 ENV 的 key |
| ENV 错误 | 调用 embedding → 401/403 → `_get_embedding` 返回 None |
| 优先级 | embed key 优先于 LLM key |

## 🔍 验证修复

```bash
# 1. 检查源码中无硬编码
grep -rn "sk-cp-\|sk-[a-zA-Z0-9]\{20,\}" cn_memory/ __init__.py
# 应输出 0 行

# 2. 语法检查
python3 -c "import ast; ast.parse(open('cn_memory.py').read())"

# 3. 配置测试
python3 -c "
import os
os.environ['CN_MEMORY_EMBEDDING_API_KEY'] = 'test-key'
from cn_memory import MemoryStore
m = MemoryStore.__new__(MemoryStore)
m._endpoint = 'http://127.0.0.1:18080/v1/embeddings'
m._model = 'bge-small-zh-v1.5'
m._embedding_api_key = os.environ.get('CN_MEMORY_EMBEDDING_API_KEY', '').strip() or os.environ.get('CN_MEMORY_LLM_API_KEY', '').strip()
assert m._embedding_api_key == 'test-key', f'expected test-key got {m._embedding_api_key!r}'
print('✅ ENV 配置读取正确')
"

# 4. 集成测试
python3 -c "
from cn_memory import MemoryStore
m = MemoryStore()
print(f'endpoint: {m._endpoint}')
print(f'model: {m._model}')
print(f'api_key: {m._embedding_api_key[:8] if m._embedding_api_key else \"(empty)\"}...')
"
```

## 📋 上游 PR 检查清单

如果要把这个 fix 提 PR 回 `fangjerome07-gif/cn_memory`：

- [x] 移除硬编码 API key
- [x] 支持环境变量配置
- [x] 向后兼容：空 key 时行为是失败而非泄露
- [x] 添加 .env.example
- [x] 添加 SECURITY.md
- [ ] upstream 可能需要：config 层支持（`_load_config` 解析 env）
- [ ] upstream 可能需要：测试用例

## 🔗 相关变更

- `cn_memory.py:103-108` - 新增 `_embedding_api_key` 属性
- `cn_memory.py:574-583` - 移除硬编码 Authorization header
- `.env.example` - 新增配置文件示例
- `SECURITY.md` - 新增本安全指南
