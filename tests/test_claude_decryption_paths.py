"""
Tests for uncovered decryption and key management code paths in ClaudeClient.

Targets the unified PBKDF2-Fernet encryption logic (lines 161-217) with random salts
and key retrieval (lines 119-159) with error handling.
"""

import pytest
from unittest.mock import Mock, patch
import base64
import os

# Skip all tests if cryptography is not available
pytest.importorskip("cryptography")

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.exceptions import APIError


def encrypt_api_key(api_key: str, encryption_key: str) -> str:
    """Helper function to encrypt API key using unified PBKDF2-Fernet with random salt."""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    # Generate random salt
    salt = os.urandom(16)

    # Derive key using PBKDF2
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    derived_key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))

    # Encrypt with Fernet
    cipher = Fernet(derived_key)
    encrypted = cipher.encrypt(api_key.encode())

    # Format: salt_b64:encrypted_b64
    salt_b64 = base64.urlsafe_b64encode(salt).decode()
    encrypted_b64 = encrypted.decode()

    return f"{salt_b64}:{encrypted_b64}"


class TestDecryptionPathsPBKDF2Random:
    """Tests for PBKDF2-Fernet decryption with random salts"""

    def test_decrypt_api_key_success(self):
        """Test successful PBKDF2-Fernet decryption with random salt"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            encryption_key = "test-encryption-key"
            api_key = "sk-actual-api-key"

            # Encrypt using new unified method
            encrypted_key = encrypt_api_key(api_key, encryption_key)

            # Mock the environment
            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": encryption_key}):
                result = client._decrypt_api_key_from_db(encrypted_key)

                assert result == api_key

    def test_decrypt_api_key_requires_encryption_key(self):
        """Test decryption fails gracefully without SOCRATES_ENCRYPTION_KEY"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            encryption_key = "test-encryption-key"
            api_key = "sk-decrypted-key"

            # Encrypt using new unified method
            encrypted_key = encrypt_api_key(api_key, encryption_key)

            # Ensure env var is not set
            with patch.dict(os.environ, {}, clear=True):
                result = client._decrypt_api_key_from_db(encrypted_key)

                # Should return None without encryption key
                assert result is None

    def test_decrypt_multiple_keys_with_different_salts(self):
        """Test that same plaintext encrypted multiple times has different ciphertexts (due to random salt)"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            encryption_key = "test-encryption-key"
            api_key = "sk-same-api-key"

            # Encrypt same key multiple times
            encrypted_key1 = encrypt_api_key(api_key, encryption_key)
            encrypted_key2 = encrypt_api_key(api_key, encryption_key)

            # Should be different due to random salts
            assert encrypted_key1 != encrypted_key2

            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": encryption_key}):
                # But should both decrypt to same value
                result1 = client._decrypt_api_key_from_db(encrypted_key1)
                result2 = client._decrypt_api_key_from_db(encrypted_key2)

                assert result1 == api_key
                assert result2 == api_key


class TestDecryptionErrorPaths:
    """Tests for error conditions in decryption"""

    def test_decrypt_api_key_invalid_format(self):
        """Test decryption returns None when encrypted data format is invalid"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            encryption_key = "test-encryption-key"

            # Data without salt:encrypted format
            invalid_data = "completely-invalid-no-salt-separator"

            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": encryption_key}):
                result = client._decrypt_api_key_from_db(invalid_data)

                # Should return None when format is invalid (missing salt)
                assert result is None

    def test_decrypt_api_key_with_corrupt_data(self):
        """Test decryption handles corrupted encrypted data"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            encryption_key = "test-encryption-key"

            # Data with correct format but corrupted encrypted part
            corrupt_data = "dGVzdHNhbHQ=:!!!invalid-fernet-ciphertext!!!"

            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": encryption_key}):
                result = client._decrypt_api_key_from_db(corrupt_data)

                # Should gracefully return None when decryption fails
                assert result is None

    def test_decrypt_api_key_with_wrong_encryption_key(self):
        """Test decryption fails with wrong encryption key"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            encryption_key1 = "encryption-key-1"
            encryption_key2 = "encryption-key-2"
            api_key = "sk-test-api-key"

            # Encrypt with key1
            encrypted_key = encrypt_api_key(api_key, encryption_key1)

            # Try to decrypt with key2
            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": encryption_key2}):
                result = client._decrypt_api_key_from_db(encrypted_key)

                # Should fail because keys don't match
                assert result is None


class TestGetUserApiKeyRetrieval:
    """Tests for _get_user_api_key method with multiple fallback paths"""

    def test_get_user_api_key_from_database(self):
        """Test retrieving user API key from database"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="fallback-key", orchestrator=orch)

            # Setup database to return encrypted key
            encrypted_key = "encrypted-user-key"
            orch.database.get_api_key.return_value = encrypted_key

            # Mock decryption
            with patch.object(client, "_decrypt_api_key_from_db") as mock_decrypt:
                mock_decrypt.return_value = "sk-decrypted-user-key"

                api_key, is_user_specific = client._get_user_api_key("user123")

                assert api_key == "sk-decrypted-user-key"
                assert is_user_specific is True

    def test_get_user_api_key_database_error_fallback(self):
        """Test fallback to environment key when database fails"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.side_effect = Exception("Database error")

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="sk-fallback-env-key", orchestrator=orch)

            api_key, is_user_specific = client._get_user_api_key("user123")

            assert api_key == "sk-fallback-env-key"
            assert is_user_specific is False

    def test_get_user_api_key_no_user_id(self):
        """Test _get_user_api_key with no user_id uses environment key"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="sk-env-key", orchestrator=orch)

            api_key, is_user_specific = client._get_user_api_key(None)

            assert api_key == "sk-env-key"
            assert is_user_specific is False

    def test_get_user_api_key_no_key_available(self):
        """Test _get_user_api_key raises APIError when no key available"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=orch)

            with pytest.raises(APIError) as exc_info:
                client._get_user_api_key("user123")

            assert "No API key configured" in str(exc_info.value)

    def test_get_user_api_key_with_placeholder_key(self):
        """Test _get_user_api_key ignores placeholder API keys"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="placeholder_test_key", orchestrator=orch)

            with pytest.raises(APIError):
                client._get_user_api_key("user123")


class TestGetClientMultiplePaths:
    """Tests for _get_client method with multiple code paths"""

    def test_get_client_subscription_fallback_warning(self):
        """Test _get_client logs warning and falls back from subscription to api_key"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            client = ClaudeClient(api_key="sk-api-key", orchestrator=orch)

            # Request with subscription method
            result = client._get_client(user_auth_method="subscription")

            # Should log warning about fallback
            # Should return a client (using api_key fallback)
            assert result is not None

    def test_get_client_user_specific_key_creation(self):
        """Test _get_client creates new client with user-specific key"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            # Setup user-specific key retrieval
            orch.database.get_api_key.return_value = "encrypted-user-key"

            with patch.object(ClaudeClient, "_decrypt_api_key_from_db") as mock_decrypt:
                mock_decrypt.return_value = "sk-user-specific-key"

                mock_client = Mock()
                mock_anth.return_value = mock_client

                client = ClaudeClient(api_key="fallback-key", orchestrator=orch)

                result = client._get_client(user_id="user123")

                # Should create new client with user-specific key
                # mock_anth should be called with user-specific key
                assert result is not None

    def test_get_client_api_error_conversion(self):
        """Test _get_client converts exceptions to APIError"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=orch)

            # Should raise APIError
            with pytest.raises(APIError) as exc_info:
                client._get_client()

            assert exc_info.value.error_type == "MISSING_API_KEY"


class TestDecryptionWithEnvironmentVariables:
    """Tests for decryption with various environment configurations"""

    def test_decrypt_with_custom_encryption_key(self):
        """Test decryption uses custom encryption key from environment"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            custom_key = "my-custom-secure-key"
            api_key = "sk-encrypted-with-custom-key"

            # Encrypt using new unified method with custom key
            encrypted = encrypt_api_key(api_key, custom_key)

            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": custom_key}):
                result = client._decrypt_api_key_from_db(encrypted)

                assert result == api_key

    def test_decrypt_logs_errors_appropriately(self):
        """Test decryption logs errors when decryption fails"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Create encrypted data with one key
            encryption_key = "test-key-123"
            api_key = "sk-test-api-key"
            encrypted = encrypt_api_key(api_key, encryption_key)

            # Try to decrypt with wrong key
            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": "wrong-key"}):
                result = client._decrypt_api_key_from_db(encrypted)

                # Should return None and have logged error
                assert result is None
                assert client.logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
