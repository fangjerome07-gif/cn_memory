## 🔐 移除硬编码 API key，改用环境变量配置

### 问题

fork 自 upstream 的 `cn_memory.py:575` 硬编码了一个 minimax API key：

```python
headers={"Content-Type": "application/json",
         "Authorization": "Bearer sk-cp-oS3OC6jigAUKdajOAMp8ithhi34_ZS-..."}
```

**安全风险**:
- API key 暴露在源码仓库
- 即使本地 ENV 配了新 key，旧 key 仍可被使用（fallback）
- 不可审计

### 修复

| 文件 | 改动 |
|------|------|
| `cn_memory.py:103-108` | 新增 `self._embedding_api_key` 属性，从 `CN_MEMORY_EMBEDDING_API_KEY` env 读取 |
| `cn_memory.py:574-583` | 移除硬编码 Authorization header，改为基于 `self._embedding_api_key` 动态构建 |
| `.env.example` | 新增配置文件示例 |
| `SECURITY.md` | 新增安全配置指南 |

### ENV 优先级链

```
CN_MEMORY_EMBEDDING_API_KEY  (embedding 专用，优先)
  ↓ (如果未设置)
CN_MEMORY_LLM_API_KEY  (LLM 共享，作为 fallback)
  ↓ (如果未设置)
""  (空字符串，调用时返回 401，绝不回落到硬编码)
```

### 向后兼容

- ✅ 旧 `CN_MEMORY_LLM_API_KEY` ENV 仍可工作（作为 fallback）
- ✅ 缺 key 时行为是失败 (401) 而非泄露硬编码
- ⚠️ 如果之前依赖硬编码 fallback 的部署会断 —— 但这正是我们想修的

### 测试

5/5 通过：
1. ✅ 源码无硬编码 (grep 验证)
2. ✅ Python 语法检查
3. ✅ ENV 优先级链 (4 场景)
4. ✅ Module import
5. ✅ 实际实例化 + 属性验证

测试脚本: `/tmp/test_cn_memory_security.py`

### 升级建议

```bash
# 旧部署: 硬编码 (不安全)
# → 新部署: ENV
export CN_MEMORY_EMBEDDING_API_KEY=sk-your-key
# 或
export CN_MEMORY_LLM_API_KEY=sk-your-shared-key
```

### 检查清单

- [x] 移除硬编码 API key
- [x] 支持环境变量配置
- [x] 向后兼容 (LLM key 作为 fallback)
- [x] 添加 .env.example
- [x] 添加 SECURITY.md
- [x] 5/5 测试通过
- [x] 0 个 `print()` 残留
- [x] 0 个硬编码 key 残留
