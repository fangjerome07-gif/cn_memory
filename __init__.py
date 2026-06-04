"""
cn_memory plugin for Hermes Agent.
Global Option 3: SQLite + numpy + llama-server(18080) + direct LLM extraction.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from datetime import datetime
import hashlib
import threading

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Tools definition
CN_MEMORY_STORE_SCHEMA = {
    "name": "cn_memory_store",
    "description": "Store high-value structured facts into the Chinese vector memory database.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or information to remember."},
            "category": {"type": "string", "description": "Legacy category: user_pref, project, tool, general"},
            "memory_type": {
                "type": "string",
                "description": "Structured type: preference, profile, project, rule, health, tool, relationship, general"
            },
            "importance": {"type": "integer", "description": "Importance from 1 to 5"},
            "confidence": {"type": "number", "description": "Confidence from 0 to 1"},
            "ttl_days": {"type": "number", "description": "Optional days before this memory expires"}
        },
        "required": ["content"]
    }
}

CN_MEMORY_SEARCH_SCHEMA = {
    "name": "cn_memory_search",
    "description": "Deep semantic search in the Chinese vector memory database.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."}
        },
        "required": ["query"]
    }
}

class CnMemoryProvider(MemoryProvider):
    _MAX_CACHE_SIZE = 64  # Prevent unbounded cache growth
    _MAX_EMBED_CHARS_CJK = 420
    _MAX_EMBED_CHARS_LATIN = 1400
    _MAX_EXTRACTION_SOURCE_CHARS = 6000
    _PROFILE_BLOCK_MAX_CHARS = 850
    _RULES_BLOCK_MAX_CHARS = 650
    _DEFAULT_MEMORY_CHAR_LIMIT = 2200
    _VALID_MEMORY_TYPES = {
        "preference", "profile", "project", "rule",
        "health", "tool", "relationship", "general",
    }
    _TYPE_WEIGHTS = {
        "rule": 1.0,
        "preference": 0.8,
        "profile": 0.7,
        "health": 0.7,
        "project": 0.6,
        "tool": 0.5,
        "relationship": 0.5,
        "general": 0.3,
    }
    _TYPE_ALIASES = {
        "user_pref": "preference",
        "pref": "preference",
        "personal": "profile",
        "user": "profile",
        "soul": "profile",
        "boundary": "rule",
        "boundaries": "rule",
        "policy": "rule",
        "workflow": "rule",
        "medical": "health",
        "fitness": "health",
        "exercise": "health",
        "tools": "tool",
        "relationship_pref": "relationship",
        "auto_retain": "general",
        "manual": "general",
    }

    def __init__(self):
        self._db_path = None
        self._endpoint = os.getenv("CN_MEMORY_EMBEDDING_ENDPOINT", "http://127.0.0.1:18080/v1/embeddings")
        self._model = os.getenv("CN_MEMORY_EMBEDDING_MODEL", "bge-small-zh-v1.5")
        self._cache = {}
        self._db_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._write_counter = 0
        self._hermes_home = None
        self._agent_identity = "default"
        self._memory_char_limit = self._DEFAULT_MEMORY_CHAR_LIMIT

        # Direct OpenAI-compatible LLM config for extraction.  This must not
        # point at a Hermes API-server port; doing so recursively creates full
        # agents and can cross-contaminate profiles.
        self._llm_base_url = os.getenv("CN_MEMORY_LLM_BASE_URL", "").strip()
        self._llm_model = os.getenv("CN_MEMORY_LLM_MODEL", "").strip()
        self._llm_key_env = os.getenv("CN_MEMORY_LLM_KEY_ENV", "").strip()
        self._llm_api_key = os.getenv("CN_MEMORY_LLM_API_KEY", "").strip()
        self._embedding_api_key = os.getenv("CN_MEMORY_EMBEDDING_API_KEY", "").strip() or os.getenv("CN_MEMORY_LLM_API_KEY", "").strip()
        try:
            self._llm_timeout = max(1.0, float(os.getenv("CN_MEMORY_LLM_TIMEOUT", "8")))
        except ValueError:
            self._llm_timeout = 8.0

        # Bypass system proxy for localhost requests
        self._local_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )
        self._http_opener = urllib.request.build_opener()

    def _unique_paths(self, paths: List[Path]) -> List[Path]:
        seen = set()
        unique = []
        for path in paths:
            try:
                resolved = str(path.expanduser())
            except Exception:
                resolved = str(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(path)
        return unique

    def _hermes_root_candidates(self, hermes_home: Optional[Path] = None) -> List[Path]:
        candidates = [Path.home() / ".hermes"]
        if hermes_home:
            home = Path(hermes_home)
            for path in (home, *home.parents):
                if path.name == ".hermes":
                    candidates.append(path)
                    break
        return self._unique_paths(candidates)

    def _cn_memory_config_paths(self, hermes_home: Path) -> List[Path]:
        paths = [root / "cn_memory" / "config.json" for root in self._hermes_root_candidates(hermes_home)]
        paths.append(hermes_home / "cn_memory" / "config.json")
        return self._unique_paths(paths)

    def _load_dotenv_value(self, key: str, hermes_home: Optional[Path] = None) -> str:
        if not key:
            return ""
        if os.getenv(key):
            return os.getenv(key, "").strip()
        candidates = []
        if hermes_home:
            candidates.append(hermes_home / ".env")
        candidates.extend(root / ".env" for root in self._hermes_root_candidates(hermes_home))
        for env_path in self._unique_paths(candidates):
            try:
                if not env_path.exists():
                    continue
                for raw in env_path.read_text().splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, value = line.split("=", 1)
                    if name.strip() != key:
                        continue
                    return value.strip().strip('"').strip("'")
            except Exception:
                continue
        return ""

    def _resolve_secret_value(self, value: Any, hermes_home: Optional[Path] = None) -> str:
        if not value:
            return ""
        text = str(value).strip()
        if text.startswith("${") and text.endswith("}"):
            return self._load_dotenv_value(text[2:-1], hermes_home)
        if text.startswith("$") and len(text) > 1:
            return self._load_dotenv_value(text[1:], hermes_home)
        return text

    def _read_json_config(self, path: Path) -> Dict[str, Any]:
        try:
            if path.exists():
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _read_yaml_config(self, path: Path) -> Dict[str, Any]:
        try:
            import yaml
            if path.exists():
                data = yaml.safe_load(path.read_text()) or {}
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.debug(f"cn_memory: failed to read config {path}: {e}")
        return {}

    def _safe_int(self, value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def _safe_float(self, value: Any, default: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def _load_runtime_config(self, hermes_home: Path) -> None:
        cn_config = {}
        for path in self._cn_memory_config_paths(hermes_home):
            cn_config.update(self._read_json_config(path))

        self._endpoint = str(cn_config.get("embedding_endpoint") or self._endpoint).strip()
        self._model = str(cn_config.get("embedding_model") or self._model).strip()

        config = self._read_yaml_config(hermes_home / "config.yaml")
        memory_cfg = config.get("memory") if isinstance(config.get("memory"), dict) else {}
        self._memory_char_limit = self._safe_int(
            cn_config.get("memory_char_limit")
            or os.getenv("CN_MEMORY_CHAR_LIMIT")
            or memory_cfg.get("memory_char_limit")
            or self._memory_char_limit,
            self._DEFAULT_MEMORY_CHAR_LIMIT,
            500,
            8000,
        )
        self._llm_timeout = self._safe_float(
            cn_config.get("llm_timeout")
            or os.getenv("CN_MEMORY_LLM_TIMEOUT")
            or self._llm_timeout,
            self._llm_timeout,
            1.0,
            60.0,
        )

        model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
        provider_name = str(cn_config.get("llm_provider") or model_cfg.get("provider") or "").strip()
        providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
        provider_cfg = providers.get(provider_name) if provider_name and isinstance(providers.get(provider_name), dict) else {}

        # Profile configs normally carry providers.xiaomi; default may carry
        # model.base_url directly, so check both shapes.
        base_url = (
            cn_config.get("llm_base_url")
            or self._llm_base_url
            or model_cfg.get("base_url")
            or provider_cfg.get("base_url")
        )
        model = (
            cn_config.get("llm_model")
            or self._llm_model
            or model_cfg.get("default")
            or provider_cfg.get("default_model")
        )
        key_env = (
            cn_config.get("llm_key_env")
            or self._llm_key_env
            or provider_cfg.get("key_env")
            or model_cfg.get("key_env")
        )
        if not key_env and provider_name:
            key_env = f"{provider_name.upper().replace('-', '_')}_API_KEY"

        direct_key = (
            cn_config.get("llm_api_key")
            or self._llm_api_key
            or model_cfg.get("api_key")
            or provider_cfg.get("api_key")
        )

        self._llm_base_url = str(base_url or "").strip()
        self._llm_model = str(model or "").strip()
        self._llm_key_env = str(key_env or "").strip()
        self._llm_api_key = self._resolve_secret_value(direct_key, hermes_home)
        if not self._llm_api_key and self._llm_key_env:
            self._llm_api_key = self._load_dotenv_value(self._llm_key_env, hermes_home)

    def _chat_completions_endpoint(self) -> str:
        base = (self._llm_base_url or "").rstrip("/")
        if not base:
            return ""
        if "anthropic" in base.lower():
            if base.endswith("/messages") or base.endswith("/v1/messages"):
                return base
            if base.endswith("/v1"):
                return f"{base}/messages"
            return f"{base}/v1/messages"
        else:
            if base.endswith("/chat/completions"):
                return base
            return f"{base}/chat/completions"
        
    @property
    def name(self) -> str:
        return "cn_memory"
        
    def is_available(self) -> bool:
        return True
        
    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get('hermes_home', str(Path.home() / '.hermes'))
        agent_identity = kwargs.get("agent_identity", "default")
        self._hermes_home = Path(hermes_home)
        self._agent_identity = str(agent_identity or "default")
        self._load_runtime_config(self._hermes_home)

        # Strict isolation: Database per profile
        # For named profiles, HERMES_HOME points to profiles/{name}/ so
        # db_dir = profiles/{name}/cn_memory/.  For the default profile,
        # HERMES_HOME is ~/.hermes — check if a profile-local cn_memory
        # directory exists there; if so, use it instead of the global one.
        default_profile_local = self._hermes_home / "profiles" / self._agent_identity / "cn_memory"
        db_dir = self._hermes_home / "cn_memory"
        if self._hermes_home == Path.home() / ".hermes" and default_profile_local.is_dir():
            db_dir = default_profile_local
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_dir / f"{self._agent_identity}_memory.sqlite"
        
        self._init_db()
        self._seed_write_counter()
        
    def _get_conn(self):
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _seed_write_counter(self) -> None:
        try:
            with self._get_conn() as conn:
                count = conn.execute("SELECT COUNT(*) FROM memory WHERE status='active'").fetchone()[0]
            with self._counter_lock:
                self._write_counter = int(count or 0)
        except Exception:
            pass
        
    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    category TEXT DEFAULT 'general',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    timestamp REAL,
                    memory_type TEXT DEFAULT 'general',
                    importance INTEGER DEFAULT 3,
                    confidence REAL DEFAULT 0.8,
                    source TEXT DEFAULT '',
                    last_seen REAL DEFAULT 0,
                    expires_at REAL,
                    status TEXT DEFAULT 'active'
                )
            ''')
            conn.commit()
        # Auto-migrate existing simple-schema DBs BEFORE creating indexes
        # that reference columns only present in the extended schema.
        self._ensure_extended_schema()
        self._ensure_v3_schema()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_hash ON memory(content_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_status ON memory(status)')
            conn.commit()

    def _ensure_extended_schema(self):
        """Migrate simple-schema DBs (missing content_hash/dim/metadata/created_at) to extended schema."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cols = {row[1] for row in cursor.execute("PRAGMA table_info(memory)").fetchall()}
            if 'content_hash' in cols:
                return  # Already extended
            # Add missing columns
            cursor.execute("ALTER TABLE memory ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
            cursor.execute("ALTER TABLE memory ADD COLUMN dim INTEGER NOT NULL DEFAULT 0")
            cursor.execute("ALTER TABLE memory ADD COLUMN metadata TEXT DEFAULT '{}'")
            cursor.execute("ALTER TABLE memory ADD COLUMN created_at REAL NOT NULL DEFAULT 0")
            # Backfill: populate content_hash and created_at from existing data
            now = time.time()
            for row in cursor.execute("SELECT id, content, timestamp FROM memory").fetchall():
                row_id, content, ts = row
                h = hashlib.md5(content.encode('utf-8')).hexdigest()
                created = ts if ts else now
                cursor.execute(
                    "UPDATE memory SET content_hash=?, dim=?, created_at=? WHERE id=?",
                    (h, 0, created, row_id)
                )
            # Use regular index (not UNIQUE) since backfilled rows may share default values.
            # Dedup for migrated DBs is enforced by INSERT OR IGNORE + hash check in _store.
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_hash ON memory(content_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category)')
            conn.commit()
            logger.info(f"cn_memory: migrated DB to extended schema")

    def _memory_type_from_category(self, category: str, content: str = "") -> str:
        raw = str(category or "").strip().lower()
        mapped = self._TYPE_ALIASES.get(raw, raw)
        if mapped in self._VALID_MEMORY_TYPES and mapped != "general":
            return mapped
        text = str(content or "")
        if any(token in text for token in ("不要", "必须", "禁止", "铁律", "规则", "不能")):
            return "rule"
        if mapped in self._VALID_MEMORY_TYPES:
            return mapped
        return "general"

    def _default_importance_for_type(self, memory_type: str) -> int:
        return {
            "rule": 5,
            "preference": 4,
            "profile": 4,
            "health": 4,
            "project": 4,
            "tool": 3,
            "relationship": 3,
            "general": 3,
        }.get(memory_type, 3)

    def _ensure_v3_schema(self) -> None:
        """Add structured memory and profile summary fields without touching old data."""
        now = time.time()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cols = {row[1] for row in cursor.execute("PRAGMA table_info(memory)").fetchall()}
            migrations = {
                "memory_type": "ALTER TABLE memory ADD COLUMN memory_type TEXT DEFAULT 'general'",
                "importance": "ALTER TABLE memory ADD COLUMN importance INTEGER DEFAULT 3",
                "confidence": "ALTER TABLE memory ADD COLUMN confidence REAL DEFAULT 0.8",
                "source": "ALTER TABLE memory ADD COLUMN source TEXT DEFAULT ''",
                "last_seen": "ALTER TABLE memory ADD COLUMN last_seen REAL DEFAULT 0",
                "expires_at": "ALTER TABLE memory ADD COLUMN expires_at REAL",
                "status": "ALTER TABLE memory ADD COLUMN status TEXT DEFAULT 'active'",
            }
            for col, ddl in migrations.items():
                if col not in cols:
                    cursor.execute(ddl)

            rows = cursor.execute(
                "SELECT id, content, category, COALESCE(timestamp, created_at, ?), "
                "memory_type, importance, confidence, last_seen, status FROM memory",
                (now,),
            ).fetchall()
            for row in rows:
                row_id, content, category, ts, memory_type, importance, confidence, last_seen, status = row
                mapped_mt = self._memory_type_from_category(category, content)
                if memory_type not in self._VALID_MEMORY_TYPES or (memory_type == "general" and mapped_mt != "general"):
                    mt = mapped_mt
                else:
                    mt = memory_type
                imp = self._safe_int(importance, self._default_importance_for_type(mt), 1, 5)
                conf = self._safe_float(confidence, 0.8, 0.0, 1.0)
                seen = last_seen if last_seen and last_seen > 0 else (ts or now)
                st = status if status in {"active", "archived", "superseded"} else "active"
                cursor.execute(
                    "UPDATE memory SET memory_type=?, importance=?, confidence=?, last_seen=?, status=? WHERE id=?",
                    (mt, imp, conf, seen, st, row_id),
                )

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profile_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_name TEXT NOT NULL UNIQUE,
                    summary TEXT DEFAULT '',
                    preferences TEXT DEFAULT '',
                    stable_facts TEXT DEFAULT '',
                    boundaries TEXT DEFAULT '',
                    updated_at REAL NOT NULL
                )
            ''')
            cursor.execute(
                "INSERT OR IGNORE INTO profile_summary (profile_name, updated_at) VALUES (?, ?)",
                (self._agent_identity, now),
            )
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_status ON memory(status)')
            
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(content, content_hash UNINDEXED, tokenize='trigram')
            ''')
            conn.commit()
        self._reconcile_fts()

    def _reconcile_fts(self) -> None:
        """Synchronize memory_fts with memory table: remove stale rows, insert missing active rows."""
        try:
            now = time.time()
            with self._get_conn() as conn:
                cursor = conn.cursor()
                # Remove true orphan FTS rows (memory row no longer exists)
                cursor.execute(
                    "DELETE FROM memory_fts WHERE rowid NOT IN (SELECT id FROM memory)"
                )
                # Remove FTS rows for non-active or expired memories
                cursor.execute(
                    "DELETE FROM memory_fts WHERE rowid IN ("
                    "SELECT id FROM memory WHERE status != 'active' OR "
                    "(expires_at IS NOT NULL AND expires_at < ?))",
                    (now,),
                )
                # Insert FTS rows for active, non-expired memories missing from FTS
                cursor.execute(
                    "INSERT INTO memory_fts(rowid, content, content_hash) "
                    "SELECT m.id, m.content, m.content_hash FROM memory m "
                    "LEFT JOIN memory_fts f ON m.id = f.rowid "
                    "WHERE f.rowid IS NULL AND m.status = 'active' "
                    "AND (m.expires_at IS NULL OR m.expires_at > ?)",
                    (now,),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"cn_memory FTS reconcile failed: {e}")

    def _trim_middle(self, text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(text) <= max_chars:
            return text
        head = max_chars * 2 // 3
        tail = max_chars - head
        return f"{text[:head]}\n...\n{text[-tail:]}"

    def _embedding_input(self, text: str) -> str:
        raw = str(text or "")
        cjk = sum(1 for ch in raw if "\u4e00" <= ch <= "\u9fff")
        max_chars = self._MAX_EMBED_CHARS_CJK if cjk >= max(1, len(raw) // 4) else self._MAX_EMBED_CHARS_LATIN
        return self._trim_middle(raw, max_chars)

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        prepared = self._embedding_input(text)
        if not prepared:
            return None
        try:
            headers = {"Content-Type": "application/json"}
            if self._embedding_api_key:
                headers["Authorization"] = f"Bearer {self._embedding_api_key}"
            req = urllib.request.Request(
                self._endpoint,
                data=json.dumps({"input": prepared, "model": self._model}).encode('utf-8'),
                headers=headers,
                method="POST"
            )
            with self._local_opener.open(req, timeout=3.0) as response:
                result = json.loads(response.read().decode('utf-8'))
                embedding = result["data"][0]["embedding"]
                return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.debug(f"cn_memory: Embedding failed: {e}")
            return None

    def _normalize_memory_type(self, value: Any, category: str = "", content: str = "") -> str:
        raw = str(value or "").strip().lower()
        mapped = self._TYPE_ALIASES.get(raw, raw)
        if mapped in self._VALID_MEMORY_TYPES:
            return mapped
        return self._memory_type_from_category(category, content)

    def _normalize_fact(self, fact: Any, category: str = "general") -> Optional[Dict[str, Any]]:
        now = time.time()
        if isinstance(fact, str):
            content = fact.strip()
            raw = {}
        elif isinstance(fact, dict):
            raw = fact
            content = str(
                raw.get("content")
                or raw.get("fact")
                or raw.get("text")
                or raw.get("memory")
                or ""
            ).strip()
        else:
            return None

        if not content:
            return None

        memory_type = self._normalize_memory_type(
            raw.get("memory_type") or raw.get("type") or raw.get("category"),
            category,
            content,
        )
        importance = self._safe_int(
            raw.get("importance"),
            self._default_importance_for_type(memory_type),
            1,
            5,
        )
        confidence = self._safe_float(raw.get("confidence"), 0.8, 0.0, 1.0)
        source = str(raw.get("source") or raw.get("reason") or "").strip()

        expires_at = raw.get("expires_at")
        ttl_days = raw.get("ttl_days")
        if expires_at in ("", None) and ttl_days not in ("", None):
            try:
                ttl = float(ttl_days)
                if ttl > 0:
                    expires_at = now + ttl * 86400
            except (TypeError, ValueError):
                expires_at = None
        try:
            expires_at = float(expires_at) if expires_at not in ("", None) else None
        except (TypeError, ValueError):
            expires_at = None

        raw_meta = raw.get("metadata")
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except (json.JSONDecodeError, ValueError):
                raw_meta = {}
        if not isinstance(raw_meta, dict):
            raw_meta = {}

        return {
            "content": content,
            "category": str(raw.get("category") or category or "general").strip() or "general",
            "memory_type": memory_type,
            "importance": importance,
            "confidence": confidence,
            "source": source,
            "expires_at": expires_at,
            "metadata": {
                "profile": self._agent_identity,
                "raw_type": str(raw.get("type") or raw.get("memory_type") or ""),
                "entities": raw_meta.get("entities", []),
            },
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _active_expiry_clause(self) -> str:
        return "status='active' AND (expires_at IS NULL OR expires_at > ?)"

    def _search(self, query: str, limit: int = 5, min_score: float = 0.45) -> List[Dict[str, Any]]:
        query_emb = self._get_embedding(query)
        has_emb = query_emb is not None
        now = time.time()

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, content, embedding, COALESCE(timestamp, created_at), "
                "memory_type, importance, confidence, last_seen FROM memory "
                f"WHERE {self._active_expiry_clause()}",
                (now,),
            )
            rows = cursor.fetchall()

        if not rows:
            return []

        valid_ids, valid_contents, valid_timestamps, valid_types = [], [], [], []
        valid_importance, valid_confidence, valid_last_seen = [], [], []
        vectors = []
        vec_indices = []  # maps vector index -> valid_ids index
        for row_id, content, blob, ts, memory_type, importance, confidence, last_seen in rows:
            vi = len(valid_ids)
            valid_ids.append(row_id)
            valid_contents.append(content)
            valid_timestamps.append(ts)
            valid_types.append(memory_type or "general")
            valid_importance.append(self._safe_int(importance, 3, 1, 5))
            valid_confidence.append(self._safe_float(confidence, 0.8, 0.0, 1.0))
            valid_last_seen.append(last_seen or ts or now)
            if has_emb:
                vec = np.frombuffer(blob, dtype=np.float32)
                if vec.shape == query_emb.shape:
                    vectors.append(vec)
                    vec_indices.append(vi)

        if not valid_ids:
            return []

        ids = valid_ids
        contents = valid_contents
        timestamps = valid_timestamps

        similarities = np.zeros(len(valid_ids), dtype=np.float32)
        if has_emb and vectors:
            matrix = np.vstack(vectors)
            q_norm = np.linalg.norm(query_emb)
            if q_norm > 0:
                query_unit = query_emb / q_norm
                row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                row_norms[row_norms == 0] = 1.0
                matrix_unit = matrix / row_norms
                vec_sims = matrix_unit @ query_unit
                for k, vi in enumerate(vec_indices):
                    similarities[vi] = vec_sims[k]

        bm25_scores = {}
        like_bonus = {}
        safe_query = query.replace('"', ' ').replace("'", ' ').strip()
        tokens = [t for t in safe_query.split() if t]

        def _escape_like(s: str) -> str:
            return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        # For CJK queries, generate n-gram fragments for better FTS recall
        cjk_chars = [ch for ch in safe_query if "一" <= ch <= "鿿"]
        cjk_fragments = []
        if len(cjk_chars) >= 2:
            cjk_text = "".join(cjk_chars)
            for n in (3, 2):
                for start in range(len(cjk_text) - n + 1):
                    cjk_fragments.append(cjk_text[start:start + n])

        try:
            with self._get_conn() as conn:
                if tokens:
                    fts_query = " AND ".join(f'"{t}"' for t in tokens)
                    for r_id, b_score in conn.execute(
                        "SELECT rowid, bm25(memory_fts) FROM memory_fts WHERE memory_fts MATCH ? LIMIT 100",
                        (fts_query,)
                    ):
                        bm25_scores[r_id] = abs(b_score)
                # Fallback: if AND query returned nothing and we have CJK fragments, try OR
                if not bm25_scores and cjk_fragments:
                    or_query = " OR ".join(f'"{f}"' for f in cjk_fragments[:8])
                    try:
                        for r_id, b_score in conn.execute(
                            "SELECT rowid, bm25(memory_fts) FROM memory_fts WHERE memory_fts MATCH ? LIMIT 100",
                            (or_query,)
                        ):
                            bm25_scores[r_id] = max(bm25_scores.get(r_id, 0), abs(b_score))
                    except Exception:
                        pass
                # LIKE on full query
                escaped_query = _escape_like(safe_query)
                for r_id, in conn.execute(
                    "SELECT id FROM memory WHERE content LIKE ? ESCAPE '\\' AND status='active' LIMIT 100",
                    (f"%{escaped_query}%",)
                ):
                    like_bonus[r_id] = 1.0
                # LIKE on CJK fragments if full query didn't match
                if not like_bonus and cjk_fragments:
                    for frag in cjk_fragments[:4]:
                        try:
                            escaped_frag = _escape_like(frag)
                            for r_id, in conn.execute(
                                "SELECT id FROM memory WHERE content LIKE ? ESCAPE '\\' AND status='active' LIMIT 50",
                                (f"%{escaped_frag}%",)
                            ):
                                like_bonus[r_id] = max(like_bonus.get(r_id, 0), 0.5)
                        except Exception:
                            pass
        except Exception as e:
            logger.debug(f"cn_memory FTS/LIKE search failed: {e}")

        # Rank-based BM25 normalization: best rank = 1.0, linearly decreasing
        bm25_ranks = {}
        if bm25_scores:
            sorted_ids = sorted(bm25_scores, key=bm25_scores.get, reverse=True)
            n = len(sorted_ids)
            for rank, r_id in enumerate(sorted_ids):
                bm25_ranks[r_id] = 1.0 - rank / max(n, 1) if n > 1 else 1.0

        idx = []
        for i, row_id in enumerate(ids):
            if similarities[i] >= min_score or row_id in bm25_scores or row_id in like_bonus:
                idx.append(i)

        if not idx:
            return []

        final_scores = []
        for original_idx in idx:
            sim = similarities[original_idx]
            memory_type = valid_types[original_idx] if valid_types[original_idx] in self._VALID_MEMORY_TYPES else "general"
            importance_norm = (valid_importance[original_idx] - 1) / 4.0
            age_days = max(0.0, (now - float(valid_last_seen[original_idx] or timestamps[original_idx] or now)) / 86400.0)
            recency_norm = 1.0 / (1.0 + age_days / 30.0)
            type_weight = self._TYPE_WEIGHTS.get(memory_type, self._TYPE_WEIGHTS["general"])

            row_id = ids[original_idx]
            bm25_rank = bm25_ranks.get(row_id, 0.0)
            lk_bonus = 0.2 if row_id in like_bonus and bm25_rank == 0 else 0.0

            # Composite ranking score (not a probability; can exceed 1.0)
            final_scores.append(
                float(sim) * 0.40
                + bm25_rank * 0.30
                + lk_bonus
                + importance_norm * 0.15
                + recency_norm * 0.15
                + type_weight * 0.10
            )
        final_scores = np.array(final_scores, dtype=np.float32)

        # Get top-k indices without full sort
        k = min(limit, len(final_scores))
        top_idx = np.argpartition(final_scores, -k)[-k:]
        top_idx = top_idx[np.argsort(final_scores[top_idx])[::-1]]

        results = []
        touched_ids = []
        for i in top_idx:
            original_idx = idx[i]
            memory_id = int(ids[original_idx])
            touched_ids.append(memory_id)
            results.append({
                "id": memory_id,
                "content": contents[original_idx],
                "score": float(final_scores[i]),
                "similarity": float(similarities[i]),
                "timestamp": timestamps[original_idx],
                "memory_type": valid_types[original_idx],
                "importance": valid_importance[original_idx],
                "confidence": valid_confidence[original_idx],
            })

        if touched_ids:
            try:
                placeholders = ",".join("?" for _ in touched_ids)
                with self._get_conn() as conn:
                    conn.execute(
                        f"UPDATE memory SET last_seen=? WHERE id IN ({placeholders})",
                        [now, *touched_ids],
                    )
                    conn.commit()
            except Exception:
                pass
        return results

    def _store_fact(self, fact: Any, category: str = "general") -> str:
        normalized = self._normalize_fact(fact, category=category)
        if not normalized:
            return "failed"

        content = normalized["content"]
        emb = self._get_embedding(content)
        if emb is None:
            return "failed"

        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        now = time.time()
        metadata = json.dumps(normalized.get("metadata") or {}, ensure_ascii=False)

        with self._db_lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                # Check for duplicate by hash (works for both migrated and fresh DBs)
                existing = cursor.execute(
                    "SELECT id, status FROM memory WHERE content_hash=? LIMIT 1", (content_hash,)
                ).fetchone()
                if existing:
                    existing_id, existing_status = existing
                    if existing_status != "active":
                        # Reactivate inactive (superseded/archived) memory and restore FTS
                        cursor.execute(
                            "UPDATE memory SET status='active', last_seen=?, "
                            "importance=MAX(importance, ?), confidence=MAX(confidence, ?) WHERE id=?",
                            (now, normalized["importance"], normalized["confidence"], existing_id),
                        )
                        cursor.execute(
                            "INSERT OR REPLACE INTO memory_fts(rowid, content, content_hash) VALUES (?, ?, ?)",
                            (existing_id, content, content_hash),
                        )
                    else:
                        cursor.execute(
                            "UPDATE memory SET last_seen=?, importance=MAX(importance, ?), confidence=MAX(confidence, ?) WHERE id=?",
                            (now, normalized["importance"], normalized["confidence"], existing_id),
                        )
                    conn.commit()
                    return "duplicate"

                similar_rows = cursor.execute(
                    "SELECT id, embedding FROM memory WHERE "
                    f"{self._active_expiry_clause()} "
                    "ORDER BY last_seen DESC LIMIT 300",
                    (now,),
                ).fetchall()
                superseded_ids = []
                for row_id, blob in similar_rows:
                    other = np.frombuffer(blob, dtype=np.float32)
                    sim = self._cosine_similarity(emb, other)
                    if sim > 0.98:
                        cursor.execute(
                            "UPDATE memory SET last_seen=?, importance=MAX(importance, ?), confidence=MAX(confidence, ?) WHERE id=?",
                            (now, normalized["importance"], normalized["confidence"], row_id),
                        )
                        conn.commit()
                        return "duplicate"
                    if sim > 0.92:
                        superseded_ids.append(row_id)

                if superseded_ids:
                    placeholders = ",".join("?" for _ in superseded_ids)
                    cursor.execute(
                        f"UPDATE memory SET status='superseded', last_seen=? WHERE id IN ({placeholders})",
                        [now, *superseded_ids],
                    )
                    cursor.execute(
                        f"DELETE FROM memory_fts WHERE rowid IN ({placeholders})",
                        superseded_ids,
                    )

                cursor.execute(
                    "INSERT INTO memory (content, content_hash, embedding, dim, category, metadata, created_at, timestamp, "
                    "memory_type, importance, confidence, source, last_seen, expires_at, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        content,
                        content_hash,
                        emb.tobytes(),
                        len(emb),
                        normalized["category"],
                        metadata,
                        now,
                        now,
                        normalized["memory_type"],
                        normalized["importance"],
                        normalized["confidence"],
                        normalized["source"],
                        now,
                        normalized["expires_at"],
                        "active",
                    )
                )
                row_id = cursor.lastrowid
                if row_id:
                    cursor.execute(
                        "INSERT INTO memory_fts(rowid, content, content_hash) VALUES (?, ?, ?)",
                        (row_id, content, content_hash)
                    )
                conn.commit()
        return "superseded" if superseded_ids else "inserted"

    def _store(self, content: Any, category: str = "general") -> bool:
        return self._store_fact(content, category=category) != "failed"

    def _truncate_block(self, text: str, max_chars: int) -> str:
        text = str(text or "").strip()
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n..."

    def _profile_block(self, max_chars: int) -> str:
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT summary, preferences, stable_facts, boundaries FROM profile_summary WHERE profile_name=?",
                    (self._agent_identity,),
                ).fetchone()
            if not row:
                return ""
            summary, preferences, stable_facts, boundaries = [str(x or "").strip() for x in row]
            lines = []
            if summary:
                lines.append(summary)
            if preferences:
                lines.append("Preferences: " + preferences)
            if stable_facts:
                lines.append("Stable facts: " + stable_facts)
            if boundaries:
                lines.append("Boundaries: " + boundaries)
            if not lines:
                return ""
            return self._truncate_block("## User Profile\n" + "\n".join(lines), max_chars)
        except Exception as e:
            logger.debug(f"cn_memory profile block failed: {e}")
            return ""

    def _rules_block(self, max_chars: int, limit: int = 8) -> str:
        try:
            now = time.time()
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT id, content FROM memory WHERE memory_type='rule' AND "
                    f"{self._active_expiry_clause()} "
                    "ORDER BY importance DESC, last_seen DESC, COALESCE(timestamp, created_at) DESC LIMIT ?",
                    (now, limit),
                ).fetchall()
            if not rows:
                return ""
            lines = [f"- {content}" for _, content in rows if str(content).strip()]
            return self._truncate_block("## Current Rules\n" + "\n".join(lines), max_chars)
        except Exception as e:
            logger.debug(f"cn_memory rules block failed: {e}")
            return ""

    def _memories_block(self, query: str, max_chars: int, limit: int = 5) -> str:
        try:
            results = self._search(query, limit=limit)
            if not results:
                return ""
            lines = []
            for r in results:
                if r.get("similarity", 0) < 0.45:
                    continue
                prefix = f"- [{r['score']:.2f}/{r.get('similarity', 0):.2f}]"
                meta = f"({r.get('memory_type', 'general')}, importance {r.get('importance', 3)})"
                lines.append(f"{prefix} {meta} {r['content']}")
            if not lines:
                return ""
            return self._truncate_block("## Relevant Memories\n" + "\n".join(lines), max_chars)
        except Exception as e:
            logger.debug(f"cn_memory memories block failed: {e}")
            return ""

    def _build_prefetch_block(self, query: str) -> str:
        budget = self._memory_char_limit
        blocks = []

        profile_budget = min(self._PROFILE_BLOCK_MAX_CHARS, max(0, budget // 3))
        profile = self._profile_block(profile_budget)
        if profile:
            blocks.append(profile)
            budget -= len(profile) + 2

        rules_budget = min(self._RULES_BLOCK_MAX_CHARS, max(0, budget // 2))
        rules = self._rules_block(rules_budget)
        if rules:
            blocks.append(rules)
            budget -= len(rules) + 2

        memories = self._memories_block(query, max(0, budget))
        if memories:
            blocks.append(memories)

        return "\n\n".join(blocks).strip()

    def system_prompt_block(self) -> str:
        return (
            "# CN Memory (V4)\n"
            "Active. Hybrid retrieval (vector + BM25 + time decay + importance) with entity extraction.\n"
            "Use cn_memory_store to save high-value facts explicitly.\n"
            "Use cn_memory_search to search your memory. Score is a composite ranking value, not a probability.\n"
            "Expired memories are auto-archived; duplicates are auto-superseded."
        )
        
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if session_id in self._cache and self._cache[session_id]:
            return self._cache.pop(session_id)
            
        if not query:
            return ""
            
        try:
            return self._build_prefetch_block(query)
        except Exception as e:
            logger.debug(f"cn_memory prefetch failed: {e}")
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query:
            return

        def _bg_fetch():
            try:
                block = self._build_prefetch_block(query)
                if block:
                    # Evict oldest entries if cache grows too large
                    if len(self._cache) >= self._MAX_CACHE_SIZE:
                        try:
                            oldest = next(iter(self._cache))
                            del self._cache[oldest]
                        except (StopIteration, RuntimeError):
                            pass
                    self._cache[session_id] = block
            except Exception:
                pass

        t = threading.Thread(target=_bg_fetch)
        t.daemon = True
        t.start()

    def _parse_llm_json(self, content: str) -> Any:
        content = str(content or "").strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start_candidates = [i for i in (content.find("{"), content.find("[")) if i >= 0]
            if not start_candidates:
                return None
            start = min(start_candidates)
            end = max(content.rfind("}"), content.rfind("]"))
            if end <= start:
                return None
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                return None

    def _call_llm_json(self, prompt: str, *, max_tokens: int = 700, timeout: Optional[float] = None) -> Any:
        endpoint = self._chat_completions_endpoint()
        if not (endpoint and self._llm_model and self._llm_api_key):
            logger.debug("cn_memory LLM skipped: config incomplete")
            return None
            
        is_anthropic = "messages" in endpoint or "anthropic" in (self._llm_base_url or "").lower()
        
        payload = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        headers = {"Content-Type": "application/json"}
        if is_anthropic:
            headers["x-api-key"] = self._llm_api_key
            headers["anthropic-version"] = "2023-06-01"
            headers["Authorization"] = f"Bearer {self._llm_api_key}"
        else:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"
            
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method="POST",
        )
        try:
            with self._http_opener.open(req, timeout=timeout or self._llm_timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                if is_anthropic:
                    content_blocks = result.get("content", [])
                    content = ""
                    for block in content_blocks:
                        if block.get("type") == "text":
                            content += block.get("text", "")
                    if not content and content_blocks:
                        content = content_blocks[0].get("text", "")
                else:
                    content = result["choices"][0]["message"]["content"]
                    
                return self._parse_llm_json(content)
        except Exception as e:
            logger.debug(f"cn_memory LLM call failed: {e}")
            return None

    def _extract_facts(self, user_content: str, assistant_content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        now_iso = datetime.now().isoformat()
        source = self._trim_middle(
            f"用户: {user_content}\n助手: {assistant_content}",
            self._MAX_EXTRACTION_SOURCE_CHARS,
        )
        prompt = (
            "你是一个长期记忆结构化抽取器。请从下面对话中提取值得长期记忆的新事实、用户偏好、规则、画像或重要状态变更。\n"
            "如果有明确的实体(人物/地点/项目等)，请在 metadata 的 entities 数组中列出。\n"
            "不要提取闲聊、临时问候、无实质内容的泛泛而谈，也不要保存助手自己的承诺，除非它是用户明确要求的长期规则。\n"
            "【重要】content 字段必须用第三人称描述（如\"用户\"、\"飞哥\"），不要使用第一人称\"我\"。例如：\n"
            "- 正确：\"用户名叫飞哥，身高170cm\"\n"
            "- 错误：\"我叫飞哥，身高170cm\"\n"
            "同时，请提取对话中用户明确提到的待办事件（包括未来要做、已经完成或已经取消的事情）。\n"
            "短期待办、临时计划和一次性安排只放入 todo_events，不要同时放入 facts；除非它体现长期稳定偏好/规则。\n"
            f"当前本地时间是 {now_iso}，请据此把今天、明天、下午、晚上等相对时间转换为 due_at 的 ISO 时间；无法确定则为 null。\n"
            "如果没有待办事件，返回 \"todo_events\": []。\n"
            "只输出 JSON 对象，不要解释。格式必须是：\n"
            "{\n"
            "  \"facts\": [\n"
            "    {\n"
            "      \"content\": \"具体事实\",\n"
            "      \"type\": \"preference|profile|project|rule|health|tool|relationship|general\",\n"
            "      \"importance\": 1-5,\n"
            "      \"confidence\": 0-1,\n"
            "      \"ttl_days\": null,\n"
            "      \"source\": \"极短来源说明\",\n"
            "      \"metadata\": {\"entities\": [\"Jerome\"]}\n"
            "    }\n"
            "  ],\n"
            "  \"todo_events\": [\n"
            "    {\n"
            "      \"content\": \"待办描述\",\n"
            "      \"status\": \"open|done|cancelled\",\n"
            "      \"due_at\": \"ISO时间或null\",\n"
            "      \"category\": \"fitness|health|config|errand|general\",\n"
            "      \"source\": \"极短来源说明\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "说明：\n"
            "- todo_events 的 content 请写标准化目标任务，不要写情绪/状态句；例如用户说\"训练完了\"时 content 应写\"今天训练\"或\"训练\"。\n"
            "- \"训练完了\"/\"理发去了\" -> status=done\n"
            "- \"不练了\"/\"理发取消\" -> status=cancelled\n"
            "- \"今天训练\"/\"下午理发\" -> status=open\n"
            "- 改期/推迟/提前必须输出两个事件：旧时间的同一待办 status=cancelled，新时间的同一待办 status=open。例如\"下午不理发了，改到明天\" -> cancelled: \"今天下午理发\" + open: \"明天理发\"。\n"
            "type 只能从给定枚举中选择；importance=5 表示必须长期遵守的规则/边界，1 表示低价值背景。\n\n"
            f"{source}"
        )
        try:
            parsed = self._call_llm_json(prompt, max_tokens=1200)
            facts_raw = []
            todo_events_raw = []
            if isinstance(parsed, list):
                facts_raw = parsed
            elif isinstance(parsed, dict):
                facts_list = parsed.get("facts")
                if isinstance(facts_list, list):
                    facts_raw = facts_list
                todo_list = parsed.get("todo_events")
                if isinstance(todo_list, list):
                    todo_events_raw = todo_list
            
            facts = []
            for item in facts_raw:
                normalized = self._normalize_fact(item, category="auto_retain")
                if normalized and len(normalized["content"]) > 3:
                    facts.append(normalized)
            
            todo_events = []
            for item in todo_events_raw:
                if isinstance(item, dict):
                    todo_events.append(item)
                    
            return facts, todo_events
        except Exception as e:
            logger.debug(f"cn_memory extraction failed: {e}")
            return [], []

    def _process_todo_events(self, todo_events: List[Dict]) -> None:
        """处理待办事件：读取旧状态，匹配完成/取消，全量覆写当前快照。"""
        if not todo_events:
            return
        
        todos_dir = Path.home() / ".hermes" / "shared" / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)
        json_path = todos_dir / f"{self._agent_identity}_todos.json"
        tmp_path = json_path.with_suffix(".json.tmp")
        
        # 1. 读取旧状态
        existing = []
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text())
            except Exception:
                existing = []
        
        # 2. 索引旧的 open todos（用于匹配 done/cancelled）
        open_todos = {t["id"]: t for t in existing if t.get("status") == "open"}
        
        # 3. 处理新事件
        now = datetime.now().isoformat()
        ordered_events = sorted(
            todo_events,
            key=lambda e: 0 if e.get("status") in ("done", "cancelled") else 1,
        )
        for event in ordered_events:
            status = event.get("status", "open")
            content = event.get("content", "").strip()
            if not content or len(content) < 2:
                continue
            
            if status == "open":
                # 检查是否和已有 open todo 语义重复（简单字符串匹配）
                is_dup = False
                for tid, todo in open_todos.items():
                    if self._todo_similar(content, todo.get("content", "")):
                        is_dup = True
                        break
                if not is_dup:
                    new_todo = {
                        "id": f"todo_{int(time.time()*1000)}_{hash(content) % 10000}",
                        "content": content,
                        "status": "open",
                        "created_at": now,
                        "due_at": event.get("due_at"),
                        "category": event.get("category", "general"),
                        "source": event.get("source", "auto"),
                    }
                    existing.append(new_todo)
                    open_todos[new_todo["id"]] = new_todo
            
            elif status in ("done", "cancelled"):
                # 尝试匹配已有的 open todo
                matched_id = None
                best_score = 0
                for tid, todo in open_todos.items():
                    score = self._todo_match_score(content, todo.get("content", ""))
                    if score > best_score and score > 0.3:
                        best_score = score
                        matched_id = tid
                if matched_id:
                    for todo in existing:
                        if todo.get("id") == matched_id:
                            todo["status"] = status
                            todo["updated_at"] = now
                            break
                    open_todos.pop(matched_id, None)
        
        # 4. 只保留最近100条记录（避免文件无限增长）
        if len(existing) > 100:
            # 保留所有 open，其余按时间倒序取最近的
            open_items = [t for t in existing if t.get("status") == "open"]
            other_items = [t for t in existing if t.get("status") != "open"]
            other_items.sort(key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)
            existing = open_items + other_items[:100 - len(open_items)]
        
        # 5. 原子写入
        tmp_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
        tmp_path.rename(json_path)

    def _todo_similar(self, a: str, b: str) -> bool:
        """简单判断两个待办是否语义重复。"""
        a_clean = a.lower().strip()
        b_clean = b.lower().strip()
        if a_clean == b_clean:
            return True
        # 包含关系
        if a_clean in b_clean or b_clean in a_clean:
            return True
        return False

    def _todo_match_score(self, event_text: str, todo_content: str) -> float:
        """计算待办事件和已有 todo 的匹配分数。"""
        event_norm = re.sub(r"\s+", "", str(event_text or "").lower())
        todo_norm = re.sub(r"\s+", "", str(todo_content or "").lower())
        event_norm = re.sub(r"[，。！？、,.!?:：；;\"'（）()【】\\[\\]{}<>《》]", "", event_norm)
        todo_norm = re.sub(r"[，。！？、,.!?:：；;\"'（）()【】\\[\\]{}<>《》]", "", todo_norm)
        if not event_norm or not todo_norm:
            return 0.0
        if event_norm == todo_norm or event_norm in todo_norm or todo_norm in event_norm:
            return 1.0

        # Chinese todo confirmations often differ only by completion/cancel
        # suffixes, e.g. "训练完了" vs "今天训练".
        stop_chars = set("今天明昨上下午夜晚早晨中后点分周星期礼拜号了完去过已经已不取消先暂回头一下要把的个些这那")
        event_chars = {
            ch for ch in event_norm
            if ch not in stop_chars and ("\u4e00" <= ch <= "\u9fff" or ch.isalnum())
        }
        todo_chars = {
            ch for ch in todo_norm
            if ch not in stop_chars and ("\u4e00" <= ch <= "\u9fff" or ch.isalnum())
        }
        if event_chars and todo_chars:
            return len(event_chars & todo_chars) / max(len(event_chars), len(todo_chars))

        event_words = set(event_norm.split())
        todo_words = set(todo_norm.split())
        if not event_words or not todo_words:
            return 0.0
        overlap = event_words & todo_words
        return len(overlap) / max(len(event_words), len(todo_words))

    def _record_writes(self, count: int) -> List[str]:
        if count <= 0:
            return []
        triggers = []
        with self._counter_lock:
            before = self._write_counter
            self._write_counter += count
            after = self._write_counter
        if before // 20 != after // 20:
            triggers.append("dedupe")
        if before // 50 != after // 50:
            triggers.append("profile")
        if "profile" not in triggers and self._profile_summary_empty():
            triggers.append("profile")
        return triggers

    def _profile_summary_empty(self) -> bool:
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT summary, preferences, stable_facts, boundaries FROM profile_summary WHERE profile_name=?",
                    (self._agent_identity,),
                ).fetchone()
            return not row or not any(str(value or "").strip() for value in row)
        except Exception:
            return False

    def _dedupe_active_memories(self, max_seconds: float = 3.0) -> None:
        start = time.monotonic()
        now = time.time()
        try:
            with self._db_lock:
                with self._get_conn() as conn:
                    rows = conn.execute(
                        "SELECT id, embedding, importance, last_seen FROM memory WHERE "
                        f"{self._active_expiry_clause()} "
                        "ORDER BY importance DESC, last_seen DESC LIMIT 300",
                        (now,),
                    ).fetchall()
                    updates = []
                    vectors = []
                    for row_id, blob, importance, last_seen in rows:
                        vec = np.frombuffer(blob, dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm == 0:
                            continue
                        vectors.append((row_id, vec / norm, importance or 3, last_seen or 0))

                    for i in range(len(vectors)):
                        if time.monotonic() - start > max_seconds:
                            break
                        id_a, vec_a, imp_a, seen_a = vectors[i]
                        for j in range(i + 1, len(vectors)):
                            id_b, vec_b, imp_b, seen_b = vectors[j]
                            sim = float(np.dot(vec_a, vec_b))
                            if sim <= 0.92:
                                continue
                            # Keep the stronger or more recent row active.
                            score_a = float(imp_a) + float(seen_a or 0) / 1_000_000_000
                            score_b = float(imp_b) + float(seen_b or 0) / 1_000_000_000
                            updates.append(id_b if score_a >= score_b else id_a)
                    if updates:
                        unique_ids = sorted(set(updates))
                        placeholders = ",".join("?" for _ in unique_ids)
                        conn.execute(
                            f"UPDATE memory SET status='superseded', last_seen=? WHERE id IN ({placeholders})",
                            [now, *unique_ids],
                        )
                        conn.execute(
                            f"DELETE FROM memory_fts WHERE rowid IN ({placeholders})",
                            unique_ids,
                        )
                        conn.commit()
        except Exception as e:
            logger.debug(f"cn_memory maintenance dedupe failed: {e}")

    def _refresh_profile_summary(self) -> None:
        now = time.time()
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT content, memory_type, importance FROM memory WHERE "
                    f"{self._active_expiry_clause()} "
                    "ORDER BY importance DESC, last_seen DESC LIMIT 80",
                    (now,),
                ).fetchall()
            if not rows:
                return
            memory_lines = [
                f"- ({memory_type}, importance {importance}) {content}"
                for content, memory_type, importance in rows
            ]
            memory_text = self._trim_middle("\n".join(memory_lines), 6000)
            prompt = (
                f"你是 {self._agent_identity} 实例的长期用户画像整理器。请根据下面结构化记忆，整理一份短画像。\n"
                "只输出 JSON 对象，不要解释。格式：\n"
                "{\"summary\":\"一句话画像\",\"preferences\":\"用户偏好，短句；分号隔开\",\"stable_facts\":\"稳定事实，短句；分号隔开\",\"boundaries\":\"必须遵守的边界/规则，短句；分号隔开\"}\n"
                "要求：只保留长期稳定、对后续回答有帮助的信息；不要编造；总长度尽量控制在中文 800 字以内。\n\n"
                f"{memory_text}"
            )
            profile_timeout = max(8.0, self._llm_timeout)
            parsed = self._call_llm_json(prompt, max_tokens=900, timeout=profile_timeout)
            if not isinstance(parsed, dict):
                return
            summary = self._truncate_block(str(parsed.get("summary") or ""), 500)
            preferences = self._truncate_block(str(parsed.get("preferences") or ""), 900)
            stable_facts = self._truncate_block(str(parsed.get("stable_facts") or ""), 900)
            boundaries = self._truncate_block(str(parsed.get("boundaries") or ""), 900)
            if not any([summary, preferences, stable_facts, boundaries]):
                return
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO profile_summary (profile_name, summary, preferences, stable_facts, boundaries, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(profile_name) DO UPDATE SET "
                    "summary=excluded.summary, preferences=excluded.preferences, stable_facts=excluded.stable_facts, "
                    "boundaries=excluded.boundaries, updated_at=excluded.updated_at",
                    (self._agent_identity, summary, preferences, stable_facts, boundaries, now),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"cn_memory profile refresh failed: {e}")

    def _cleanup_expired_memories(self) -> None:
        now = time.time()
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                rows = cursor.execute(
                    "SELECT id FROM memory WHERE status='active' AND expires_at IS NOT NULL AND expires_at < ?",
                    (now,)
                ).fetchall()
                if rows:
                    ids = [r[0] for r in rows]
                    placeholders = ",".join("?" for _ in ids)
                    cursor.execute(f"UPDATE memory SET status='archived' WHERE id IN ({placeholders})", ids)
                    cursor.execute(f"DELETE FROM memory_fts WHERE rowid IN ({placeholders})", ids)
                    conn.commit()
        except Exception as e:
            logger.debug(f"cn_memory cleanup failed: {e}")

    def _run_maintenance(self, triggers: List[str]) -> None:
        self._cleanup_expired_memories()
        self._reconcile_fts()
        if not triggers:
            return
        if "dedupe" in triggers:
            self._dedupe_active_memories(max_seconds=3.0)
        if "profile" in triggers:
            self._refresh_profile_summary()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages=None) -> None:
        if not user_content:
            return

        def _bg_process():
            try:
                # 1. Automatic extraction via direct OpenAI-compatible LLM API
                facts, todo_events = self._extract_facts(user_content, assistant_content)
                stored = 0
                new_writes = 0
                for fact in facts:
                    action = self._store_fact(fact, category="auto_retain")
                    if action != "failed":
                        stored += 1
                    if action in {"inserted", "superseded"}:
                        new_writes += 1
                if facts:
                    logger.debug(f"cn_memory sync_turn: extracted {len(facts)} facts, stored {stored}")
                
                # 新增：处理待办事件
                if todo_events:
                    self._process_todo_events(todo_events)
                    logger.debug(f"cn_memory sync_turn: processed {len(todo_events)} todo events")

                self._run_maintenance(self._record_writes(new_writes))
            except Exception as e:
                logger.warning(f"cn_memory sync_turn failed: {e}")
                try:
                    import time as _time
                    _err_path = Path(self._hermes_home or Path.home() / '.hermes') / 'cn_memory' / 'last_error.json'
                    _err_path.parent.mkdir(parents=True, exist_ok=True)
                    _err_path.write_text(json.dumps({
                        'time': _time.strftime('%Y-%m-%d %H:%M:%S'),
                        'timestamp': _time.time(),
                        'instance': self._agent_identity,
                        'error': str(e)[:500],
                        'source': 'sync_turn'
                    }, ensure_ascii=False))
                except Exception:
                    pass

        t = threading.Thread(target=_bg_process)
        t.daemon = True
        t.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [CN_MEMORY_STORE_SCHEMA, CN_MEMORY_SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "cn_memory_store":
                action = self._store_fact(args, args.get("category", "manual"))
                if action in {"inserted", "superseded"}:
                    self._run_maintenance(self._record_writes(1))
                return json.dumps({"status": "success" if action != "failed" else "failed_to_embed", "action": action})
            elif tool_name == "cn_memory_search":
                results = self._search(args.get("query", ""))
                return json.dumps({"results": results}, ensure_ascii=False)
            else:
                return tool_error(f"Unknown cn_memory tool: {tool_name}")
        except Exception as e:
            return tool_error(str(e))

def register(ctx) -> None:
    ctx.register_memory_provider(CnMemoryProvider())
