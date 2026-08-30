"""AES-256-GCM encryption for the exported ONNX models.

Format of an encrypted blob:  MAGIC(5) || nonce(12) || ciphertext+tag
GCM authenticates the ciphertext, so a wrong key or a tampered file fails to
decrypt instead of silently returning garbage. The model is only ever decrypted
into memory and handed to onnxruntime as bytes; the plaintext ONNX never touches
disk on the deployed side.
"""

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"DDEC1"
NONCE_LEN = 12
KEY_LEN = 32  # AES-256


def generate_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def encrypt(data: bytes, key: bytes) -> bytes:
    nonce = os.urandom(NONCE_LEN)
    ciphertext = AESGCM(key).encrypt(nonce, data, None)
    return MAGIC + nonce + ciphertext


def decrypt(blob: bytes, key: bytes) -> bytes:
    if blob[:len(MAGIC)] != MAGIC:
        raise ValueError("not a DDEC1 encrypted model")
    nonce = blob[len(MAGIC):len(MAGIC) + NONCE_LEN]
    ciphertext = blob[len(MAGIC) + NONCE_LEN:]
    return AESGCM(key).decrypt(nonce, ciphertext, None)


def save_key(path: str, key: bytes) -> None:
    """Store the key base64-encoded so it survives text-mode handling."""
    with open(path, "w", encoding="ascii") as f:
        f.write(base64.b64encode(key).decode("ascii"))


def load_key(path: str) -> bytes:
    with open(path, "r", encoding="ascii") as f:
        key = base64.b64decode(f.read().strip())
    if len(key) != KEY_LEN:
        raise ValueError(f"expected a {KEY_LEN}-byte key, got {len(key)} bytes")
    return key
