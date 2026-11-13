"""Compact in-memory tables, built incrementally from the shipped sources."""

import json
import marshal
import struct
from array import array
from collections.abc import Mapping
from functools import lru_cache


class PackedMap(Mapping):
    def __init__(self, decode=bytes, width=None):
        self._keys = bytearray()
        self._values = bytearray()
        self._offsets = array("I", [0])
        self._starts = array("I")
        self._ends = array("I")
        self._index = array("I", [0]) * 16
        self._decode = decode
        self._width = width

    def __len__(self):
        return len(self._offsets) - 1

    def __iter__(self):
        for i in range(len(self)):
            yield self._keys[self._offsets[i]:self._offsets[i + 1]].decode("utf-8")

    def _slot(self, key):
        if not isinstance(key, str):
            hash(key)
            raise KeyError(key)
        slot = hash(key) & (len(self._index) - 1)
        if not self._index[slot]:
            return slot
        raw = key.encode("utf-8")
        while number := self._index[slot]:
            i = number - 1
            if self._keys[self._offsets[i]:self._offsets[i + 1]] == raw:
                return slot
            slot = (slot + 1) & (len(self._index) - 1)
        return slot

    def __getitem__(self, key):
        i = self._index[self._slot(key)] - 1
        if i < 0:
            raise KeyError(key)
        return self._value(i)

    def _value(self, i):
        start, end = (self._starts[i], self._ends[i]) if self._width is None else (i * self._width, (i + 1) * self._width)
        return self._decode(self._values[start:end])

    def get(self, key, default=None):
        try:
            i = self._index[self._slot(key)] - 1
        except KeyError:
            return default
        return default if i < 0 else self._value(i)

    def __contains__(self, key):
        try:
            return bool(self._index[self._slot(key)])
        except KeyError:
            return False

    def add(self, key, value):
        slot = self._slot(key)
        i = self._index[slot] - 1
        if i < 0:
            if (len(self) + 1) * 3 > len(self._index) * 2:
                self._index = array("I", [0]) * (len(self._index) * 2)
                for j, old in enumerate(self):
                    self._index[self._slot(old)] = j + 1
                slot = self._slot(key)
            i = len(self)
            self._index[slot] = i + 1
            self._keys.extend(key.encode("utf-8"))
            self._offsets.append(len(self._keys))
            if self._width is None:
                self._starts.append(0)
                self._ends.append(0)
        if self._width is None:
            self._starts[i] = len(self._values)
            self._values.extend(value)
            self._ends[i] = len(self._values)
        else:
            if len(value) != self._width:
                raise ValueError("inconsistent record width")
            self._values[i * self._width:(i + 1) * self._width] = value


def _doubles(raw):
    return list(struct.unpack(f"<{len(raw) // 8}d", raw))


def pack_weights(record):
    weights = record.get("weights", {})
    if len(weights) > 16:
        width = len(next(iter(weights.values())))
        packed = PackedMap(_doubles, width * 8 if all(len(v) == width for v in weights.values()) else None)
        for key, values in weights.items():
            packed.add(key, struct.pack(f"<{len(values)}d", *values))
        record["weights"] = packed
    return record


class _JSONStream:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.pos = 0
        self.decoder = json.JSONDecoder()

    def _more(self):
        chunk = self.stream.read(65536)
        self.buffer = self.buffer[self.pos:] + chunk
        self.pos = 0
        return bool(chunk)

    def _peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if not self._more():
                raise ValueError("unexpected end of JSON")

    def _expect(self, char):
        if self._peek() != char:
            raise ValueError(f"expected {char!r}")
        self.pos += 1

    def value(self):
        self._peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                if isinstance(value, (int, float)) and end < len(self.buffer) and self.buffer[end] not in " \t\r\n,}]":
                    if self._more():
                        continue
                    raise ValueError("invalid JSON number")
                if end == len(self.buffer):
                    length = end - self.pos
                    if self._more():
                        continue
                    end = length
                self.pos = end
                return value
            except json.JSONDecodeError:
                if not self._more():
                    raise

    def items(self, readers=None):
        self._expect("{")
        if self._peek() != "}":
            while True:
                key = self.value()
                self._expect(":")
                yield key, readers[key](self) if readers and key in readers else self.value()
                if self._peek() != ",":
                    break
                self.pos += 1
        self._expect("}")


def load_context(stream):
    return dict(_JSONStream(stream).items({
        "words": lambda reader: {key: pack_weights(record) for key, record in reader.items()},
    }))


def load_letters(stream):
    def contexts(reader):
        packed = PackedMap(marshal.loads)
        for key, counts in reader.items():
            packed.add(key, marshal.dumps(counts))
        return _CachedMap(packed)

    return dict(_JSONStream(stream).items({"contexts": contexts}))


class _CachedMap(Mapping):
    def __init__(self, data):
        self._data = data
        self._get = lru_cache(maxsize=4096)(data.__getitem__)

    def __getitem__(self, key):
        return self._get(key)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)
