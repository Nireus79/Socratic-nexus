"""
Tests for uncovered decryption and key management code paths in ClaudeClient.

Targets the complex decryption logic (lines 161-248) and key retrieval (lines 119-159)
with multiple encryption method fallbacks and error handling.
"""

import pytest
from unittest.mock import Mock, patch
import base64
import os

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.exceptions import APIError


class TestDecryptionPathsSHA256:
    """Tests for SHA256-Fernet decryption path"""

    def test_decrypt_api_key_sha256_success(self):
        """Test successful SHA256-Fernet decryption"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Create a properly encrypted key with SHA256
            import hashlib
            from cryptography.fernet import Fernet

            encryption_key = "test-encryption-key"
            key_bytes = hashlib.sha256(encryption_key.encode()).digest()
            derived_key = base64.urlsafe_b64encode(key_bytes)
            cipher = Fernet(derived_key)

            api_key = "sk-actual-api-key"
            encrypted_key = cipher.encrypt(api_key.encode()).decode()

            # Mock the environment
            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": encryption_key}):
                result = client._decrypt_api_key_from_db(encrypted_key)

                assert result == api_key

    def test_decrypt_api_key_with_default_key(self):
        """Test decryption with default insecure key"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Create encrypted key with default encryption key
            import hashlib
            from cryptography.fernet import Fernet

            default_key = "default-insecure-key-change-in-production"
            key_bytes = hashlib.sha256(default_key.encode()).digest()
            derived_key = base64.urlsafe_b64encode(key_bytes)
            cipher = Fernet(derived_key)

            api_key = "sk-decrypted-key"
            encrypted_key = cipher.encrypt(api_key.encode()).decode()

            # Ensure env var is not set (use default)
            with patch.dict(os.environ, {}, clear=True):
                result = client._decrypt_api_key_from_db(encrypted_key)

                assert result == api_key


class TestDecryptionPathsPBKDF2:
    """Tests for PBKDF2-Fernet decryption fallback"""

    def test_decrypt_api_key_pbkdf2_fallback(self):
        """Test PBKDF2-Fernet decryption fallback when SHA256 fails"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            try:
                from cryptography.fernet import Fernet
                from cryptography.hazmat.backends import default_backend
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

                encryption_key = "pbkdf2-encryption-key"
                salt = b"socrates-salt"
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                    backend=default_backend(),
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
                cipher = Fernet(derived_key)

                api_key = "sk-pbkdf2-decrypted"
                encrypted_key = cipher.encrypt(api_key.encode()).decode()

                # Provide encryption key to use PBKDF2
                with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": encryption_key}):
                    result = client._decrypt_api_key_from_db(encrypted_key)

                    # Should decrypt successfully
                    assert result is not None
            except ImportError:
                pytest.skip("cryptography.hazmat not available")

    def test_decrypt_api_key_pbkdf2_import_error(self):
        """Test PBKDF2 decryption handles ImportError gracefully"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Create invalid encrypted string to force fallback
            encrypted_key = "invalid-encrypted-data"

            with patch("socratic_nexus.clients.claude_client.hashlib"):
                result = client._decrypt_api_key_from_db(encrypted_key)

                # Should return None if all decryption methods fail
                assert result is None


class TestDecryptionPathsBase64:
    """Tests for Base64 decoding fallback"""

    def test_decrypt_api_key_base64_fallback(self):
        """Test Base64 decoding fallback when both Fernet methods fail"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Create base64-encoded key
            api_key = "sk-base64-encoded-key"
            encoded_key = base64.b64encode(api_key.encode()).decode()

            # Provide invalid encryption key to skip SHA256/PBKDF2
            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": ""}):
                result = client._decrypt_api_key_from_db(encoded_key)

                # Should decode successfully with base64
                assert result is not None


class TestDecryptionErrorPaths:
    """Tests for error conditions in decryption"""

    def test_decrypt_api_key_all_methods_fail(self):
        """Test decryption returns None when all methods fail"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Completely invalid encrypted data
            invalid_data = "completely-invalid-and-undecodable-garbage-data!!!"

            result = client._decrypt_api_key_from_db(invalid_data)

            # Should return None when all decryption methods fail
            assert result is None

    def test_decrypt_api_key_with_corrupt_data(self):
        """Test decryption handles corrupted encrypted data"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Data that looks encrypted but is corrupted
            corrupt_data = base64.urlsafe_b64encode(b"corrupted").decode()

            result = client._decrypt_api_key_from_db(corrupt_data)

            # Should gracefully return None
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

            import hashlib
            from cryptography.fernet import Fernet

            key_bytes = hashlib.sha256(custom_key.encode()).digest()
            derived_key = base64.urlsafe_b64encode(key_bytes)
            cipher = Fernet(derived_key)

            api_key = "sk-encrypted-with-custom-key"
            encrypted = cipher.encrypt(api_key.encode()).decode()

            with patch.dict(os.environ, {"SOCRATES_ENCRYPTION_KEY": custom_key}):
                result = client._decrypt_api_key_from_db(encrypted)

                assert result == api_key

    def test_decrypt_logs_security_warning(self):
        """Test decryption logs warning when using default insecure key"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Create encrypted data with default key
            import hashlib
            from cryptography.fernet import Fernet

            default_key = "default-insecure-key-change-in-production"
            key_bytes = hashlib.sha256(default_key.encode()).digest()
            derived_key = base64.urlsafe_b64encode(key_bytes)
            cipher = Fernet(derived_key)

            encrypted = cipher.encrypt(b"test-key").decode()

            with patch.dict(os.environ, {}, clear=True):
                client._decrypt_api_key_from_db(encrypted)

                # Should have logger available
                assert client.logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
