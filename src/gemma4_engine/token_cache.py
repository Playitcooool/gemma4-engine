from __future__ import annotations

import hashlib
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

CACHE_FORMAT_VERSION = 1
DEFAULT_TOKEN_CACHE_DIR = ".gemma4-cache/prefix-tokens"
_HEADER = b"G4TC"
_ENTRY_HEADER = struct.Struct("<4sII")


@dataclass(frozen=True)
class TokenCacheResult:
    token_ids: list[int]
    source: str

    @property
    def hit(self) -> bool:
        return self.source in ("memory", "disk")


class HierarchicalTokenCache:
    def __init__(
        self,
        *,
        disk_dir: str | os.PathLike[str] | None,
        max_memory_entries: int = 128,
    ) -> None:
        self.disk_dir = Path(disk_dir) if disk_dir else None
        self.max_memory_entries = max_memory_entries
        self._memory: OrderedDict[str, list[int]] = OrderedDict()

    def get_or_encode(
        self,
        *,
        key: str,
        encode,
    ) -> TokenCacheResult:
        memory_hit = self._memory.get(key)
        if memory_hit is not None:
            self._memory.move_to_end(key)
            return TokenCacheResult(list(memory_hit), "memory")

        disk_hit = self._read_disk(key)
        if disk_hit is not None:
            self._remember(key, disk_hit)
            return TokenCacheResult(list(disk_hit), "disk")

        token_ids = list(encode())
        self._remember(key, token_ids)
        self._write_disk(key, token_ids)
        return TokenCacheResult(token_ids, "miss")

    def clear_memory(self) -> None:
        self._memory.clear()

    def _remember(self, key: str, token_ids: list[int]) -> None:
        self._memory[key] = list(token_ids)
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)

    def _path(self, key: str) -> Path | None:
        if self.disk_dir is None:
            return None
        return self.disk_dir / f"{key}.g4tokens"

    def _read_disk(self, key: str) -> list[int] | None:
        path = self._path(key)
        if path is None:
            return None
        try:
            data = path.read_bytes()
            magic, version, count = _ENTRY_HEADER.unpack_from(data, 0)
            if magic != _HEADER or version != CACHE_FORMAT_VERSION:
                return None
            expected_size = _ENTRY_HEADER.size + count * 4
            if len(data) != expected_size:
                return None
            return list(struct.unpack_from(f"<{count}i", data, _ENTRY_HEADER.size))
        except (FileNotFoundError, OSError, struct.error, ValueError):
            return None

    def _write_disk(self, key: str, token_ids: list[int]) -> None:
        path = self._path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = _ENTRY_HEADER.pack(_HEADER, CACHE_FORMAT_VERSION, len(token_ids))
            if token_ids:
                payload += struct.pack(f"<{len(token_ids)}i", *[int(token) for token in token_ids])
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_bytes(payload)
            tmp_path.replace(path)
        except (OSError, OverflowError, struct.error):
            return


def token_cache_key(
    *,
    model_path: str,
    prompt_mode: str,
    text: str,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(CACHE_FORMAT_VERSION).encode("ascii"))
    digest.update(b"\0")
    digest.update(model_path.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(prompt_mode.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()
