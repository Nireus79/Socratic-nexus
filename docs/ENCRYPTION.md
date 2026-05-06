# Encryption Guide

Secure handling of API keys and sensitive data in Socratic Nexus.

## Table of Contents

- [Overview](#overview)
- [Encryption System](#encryption-system)
- [Setup and Configuration](#setup-and-configuration)
- [API Key Storage](#api-key-storage)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Socratic Nexus uses **PBKDF2-HMAC-SHA256 with Fernet symmetric encryption** to securely store and retrieve API keys and sensitive data. This approach provides strong security while maintaining simplicity and reliability.

### Key Features

- **PBKDF2 Key Derivation**: 100,000 iterations of HMAC-SHA256 with random salt
- **Fernet Symmetric Encryption**: Authenticated encryption with built-in integrity verification
- **Random Salt per Encryption**: Each encrypted value uses a unique salt, preventing rainbow table attacks
- **Self-Contained Format**: Salt is embedded with encrypted data (`salt_b64:encrypted_b64`)

---

## Encryption System

### How It Works

#### Encryption Process

1. Generate a random 16-byte salt
2. Derive encryption key using PBKDF2:
   - Algorithm: HMAC-SHA256
   - Iterations: 100,000
   - Salt: Random (step 1)
   - Output length: 32 bytes
3. Encrypt data using Fernet with derived key
4. Store as: `salt_b64:encrypted_b64`

#### Decryption Process

1. Extract salt from encrypted data (`salt_b64:encrypted_b64` format)
2. Derive the same key using PBKDF2 with extracted salt
3. Decrypt data using Fernet with derived key
4. Return plaintext

### Why This Approach?

| Feature | Benefit |
|---------|---------|
| PBKDF2 Key Derivation | Makes brute-force attacks computationally expensive |
| 100,000 Iterations | Slows down attackers without significantly impacting legitimate users |
| Random Salt | Prevents rainbow table attacks and precomputation |
| Fernet Encryption | Provides authenticated encryption (prevents tampering) |
| Self-Contained Format | No need for separate salt storage; salt travels with ciphertext |

---

## Setup and Configuration

### 1. Generate Encryption Key

Generate a secure random key for encrypting sensitive data:

```bash
# Option A: Using Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Example output:
# ZkL8mN9pQ2rS3tU4vW5xY6zA1bB2cC3dD4eE5fF6g

# Option B: Using OpenSSL
openssl rand -base64 32
```

### 2. Set Environment Variable

Store the encryption key in your environment:

```bash
# Linux/macOS
export SOCRATES_ENCRYPTION_KEY="ZkL8mN9pQ2rS3tU4vW5xY6zA1bB2cC3dD4eE5fF6g"

# Windows (PowerShell)
$env:SOCRATES_ENCRYPTION_KEY = "ZkL8mN9pQ2rS3tU4vW5xY6zA1bB2cC3dD4eE5fF6g"

# Windows (Command Prompt)
set SOCRATES_ENCRYPTION_KEY=ZkL8mN9pQ2rS3tU4vW5xY6zA1bB2cC3dD4eE5fF6g
```

### 3. Verify Configuration

```python
import os
from socratic_nexus.clients.claude_client import ClaudeClient

# Verify encryption key is set
encryption_key = os.getenv("SOCRATES_ENCRYPTION_KEY")
if encryption_key:
    print(f"✓ Encryption key configured ({len(encryption_key)} chars)")
else:
    print("✗ SOCRATES_ENCRYPTION_KEY not set")
```

---

## API Key Storage

### How API Keys Are Encrypted

API keys stored in the database are encrypted using the system described above:

```python
# User provides API key
api_key = "sk-ant-abc123..."

# System encrypts it (automatically)
# - Generates random salt
# - Derives key from SOCRATES_ENCRYPTION_KEY
# - Encrypts api_key with Fernet
# - Stores as: salt_b64:encrypted_b64

# System retrieves it (automatically)
# - Extracts salt from encrypted data
# - Derives same key using extracted salt
# - Decrypts with Fernet
# - Returns plaintext api_key
```

### Decryption in ClaudeClient

The `ClaudeClient` automatically handles decryption when retrieving API keys:

```python
from socratic_nexus.clients.claude_client import ClaudeClient

client = ClaudeClient(
    api_key="...",  # Default API key (can be encrypted)
    # SOCRATES_ENCRYPTION_KEY must be set for encrypted keys
)

# Internally, when database lookup occurs:
# 1. Retrieves encrypted_key from database
# 2. Calls _decrypt_api_key_from_db(encrypted_key)
# 3. Extracts salt from encrypted_key
# 4. Derives decryption key from SOCRATES_ENCRYPTION_KEY
# 5. Decrypts and returns plaintext API key
```

### Encrypted Data Format

All encrypted data follows this format:

```
salt_b64:encrypted_b64
```

- `salt_b64`: Base64-encoded 16-byte random salt
- `:`: Separator
- `encrypted_b64`: Base64-encoded Fernet-encrypted data

Example:
```
YmFzZTY0X3NhbHRfaGVyZQ==:gAAAAABmK8x5...encrypted_data_here...
```

---

## Best Practices

### 1. Key Management

**Do:**
- Generate a strong random key using `secrets.token_urlsafe(32)`
- Store key in environment variables (not in code)
- Rotate keys periodically in production
- Use different keys for different environments

**Don't:**
- Hardcode encryption keys in source code
- Use weak or predictable keys
- Share keys via insecure channels (email, chat, etc.)
- Commit keys to version control

### 2. Environment Setup

```python
# development.env
SOCRATES_ENCRYPTION_KEY=abc123_development_key_only

# production.env
SOCRATES_ENCRYPTION_KEY=xyz789_strong_production_key

# .gitignore (always include)
*.env
.env
.env.local
```

### 3. Deployment

```bash
# Docker
docker run \
  -e SOCRATES_ENCRYPTION_KEY="$ENCRYPTION_KEY" \
  my-app:latest

# Kubernetes
kubectl set env deployment/my-app \
  SOCRATES_ENCRYPTION_KEY="$ENCRYPTION_KEY"

# AWS Lambda
aws lambda update-function-configuration \
  --function-name my-function \
  --environment Variables={SOCRATES_ENCRYPTION_KEY=value}
```

### 4. Key Rotation

When rotating keys:

```python
# 1. Decrypt all keys with old key
# 2. Re-encrypt with new key
# 3. Update SOCRATES_ENCRYPTION_KEY environment variable

# For Socratic Nexus integration:
# - All keys are automatically re-encrypted on next access
# - No manual migration needed (handled transparently)
```

---

## Troubleshooting

### Common Issues

#### 1. "SOCRATES_ENCRYPTION_KEY not set"

**Error:**
```
RuntimeError: SOCRATES_ENCRYPTION_KEY environment variable is required for decryption
```

**Solution:**
```bash
# Verify key is set
echo $SOCRATES_ENCRYPTION_KEY

# If empty, set it
export SOCRATES_ENCRYPTION_KEY="your-key-here"

# Verify again
echo $SOCRATES_ENCRYPTION_KEY
```

#### 2. "Invalid encrypted data format"

**Error:**
```
ValueError: Invalid encrypted data format: missing salt
```

**Causes:**
- Data wasn't encrypted with the current system
- Data corrupted during storage/transmission
- Using incompatible encryption format

**Solution:**
- Verify encrypted data is in `salt_b64:encrypted_b64` format
- Check for data corruption during storage
- Re-encrypt data with current system if migrating

#### 3. "Failed to decrypt API key"

**Error:**
```
RuntimeError: Failed to decrypt API key: [decryption error details]
```

**Causes:**
- Wrong encryption key set
- Data encrypted with different key
- Encrypted data corrupted

**Solutions:**
```python
# Verify encryption key
import os
key = os.getenv("SOCRATES_ENCRYPTION_KEY")
print(f"Key length: {len(key) if key else 'Not set'}")

# Check if it's the right key
# (decrypt a test value encrypted with this key)

# If wrong key, update environment variable
os.environ["SOCRATES_ENCRYPTION_KEY"] = "correct-key-here"
```

#### 4. Different Results on Different Systems

**Issue:** Same encrypted data decrypts differently on different machines

**Solution:** Verify same `SOCRATES_ENCRYPTION_KEY` is set on all systems:

```bash
# System A
echo $SOCRATES_ENCRYPTION_KEY

# System B
echo $SOCRATES_ENCRYPTION_KEY

# Should be identical
```

### Debugging

Enable debug logging to troubleshoot encryption issues:

```python
import logging

# Enable debug logging for encryption operations
logging.basicConfig(level=logging.DEBUG)

# Now see detailed encryption/decryption logs
from socratic_nexus.clients.claude_client import ClaudeClient

client = ClaudeClient(api_key="...")
# Debug logs will show encryption key source and decryption method
```

### Verify Encryption Works

```python
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def test_encryption():
    """Test that encryption system is working."""

    encryption_key = os.getenv("SOCRATES_ENCRYPTION_KEY")
    if not encryption_key:
        print("✗ SOCRATES_ENCRYPTION_KEY not set")
        return False

    try:
        # Test encryption
        test_data = "sk-ant-test123"
        salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))

        cipher = Fernet(derived_key)
        encrypted = cipher.encrypt(test_data.encode())

        # Test decryption
        decrypted = cipher.decrypt(encrypted)

        if decrypted.decode() == test_data:
            print("✓ Encryption system working correctly")
            return True
        else:
            print("✗ Decryption produced different result")
            return False

    except Exception as e:
        print(f"✗ Encryption test failed: {e}")
        return False

# Run test
if test_encryption():
    print("✓ Ready to use Socratic Nexus with encrypted API keys")
else:
    print("✗ Fix encryption setup before proceeding")
```

---

## Advanced Topics

### Custom Encryption Integration

If you need custom encryption logic, you can override the decryption method:

```python
from socratic_nexus.clients.claude_client import ClaudeClient

class CustomClaudeClient(ClaudeClient):
    def _decrypt_api_key_from_db(self, encrypted_key: str):
        """Use custom decryption logic."""
        # Your custom decryption here
        return my_custom_decrypt_function(encrypted_key)

# Use custom client
client = CustomClaudeClient(api_key="...")
```

### Key Derivation Details

The system uses PBKDF2 with these parameters:

```python
{
    "algorithm": "HMAC-SHA256",
    "length": 32,
    "salt": random_16_bytes,
    "iterations": 100000,
}
```

These parameters were chosen to:
- Be resistant to GPU/ASIC attacks (100k iterations)
- Derive 256-bit keys suitable for Fernet
- Provide good performance on modern systems (< 100ms per key derivation)

---

## References

- [Fernet Specification](https://github.com/fernet/spec/blob/master/Spec.md)
- [PBKDF2 (RFC 2898)](https://tools.ietf.org/html/rfc2898)
- [Cryptography.io Documentation](https://cryptography.io/)

---

For questions or issues, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) or open an issue on GitHub.
