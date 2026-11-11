"""Unique four-position identifiers with disjoint token spans."""
import numpy as np
from ..artifacts import digest, read_json, write_json

PAD, EOS, SEP, LEFT, RIGHT = range(5)
FIRST_CODE = 5


class MaterializedIndex:
    def __init__(self, payload):
        self.payload = payload
        checksum = digest({k: v for k, v in payload.items() if k != "digest"})
        if payload.get("digest") != checksum:
            raise ValueError("Index digest mismatch")
        self.digest = checksum
        self.item_ids = payload["item_ids"]
        self.base_codes = np.asarray(payload["base_codes"], dtype=np.int64)
        self.codes = np.asarray(payload["codes"], dtype=np.int64)
        self.sizes = payload["sizes"]
        self.offsets = np.cumsum([FIRST_CODE] + self.sizes[:-1]).tolist()
        self.tokens = self.codes + np.asarray(self.offsets)
        self.vocab_size = FIRST_CODE + sum(self.sizes)
        self.inverse = {tuple(code): row for row, code in enumerate(self.tokens.tolist())}
        if len(set(self.item_ids)) != len(self.item_ids) or len(self.inverse) != len(self.item_ids):
            raise ValueError("Index must provide a bijection")
        if self.codes.shape != (len(self.item_ids), 4) or np.any(self.codes < 0) or np.any(self.codes >= self.sizes):
            raise ValueError("Invalid identifier shape or spans")
        self.support = {}
        for code in self.tokens.tolist():
            self.support.setdefault(tuple(code[:3]), set()).add(code[3])

    @classmethod
    def load(cls, path):
        return cls(read_json(path))

    def save(self, path):
        write_json(path, self.payload)

