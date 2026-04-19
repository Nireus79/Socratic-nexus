"""Tests for retry logic and exponential backoff."""

from socratic_nexus.retry import RetryConfig, exponential_backoff


def test_retry_config_defaults():
    """Test RetryConfig default values."""
    config = RetryConfig()

    assert config.max_attempts == 3
    assert config.base_delay == 1.0
    assert config.max_delay == 60.0
    assert config.exponential_base == 2.0
    assert config.jitter is True


def test_retry_config_custom():
    """Test RetryConfig with custom values."""
    config = RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        max_delay=120.0,
        exponential_base=3.0,
        jitter=False,
    )

    assert config.max_attempts == 5
    assert config.base_delay == 2.0
    assert config.max_delay == 120.0
    assert config.exponential_base == 3.0
    assert config.jitter is False


def test_exponential_backoff_attempt_0():
    """Test exponential_backoff for first attempt."""
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

    # First retry (attempt 0): 1.0 * 2^0 = 1.0
    delay = exponential_backoff(0, config)
    assert delay == 1.0


def test_exponential_backoff_attempt_1():
    """Test exponential_backoff for second attempt."""
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

    # Second retry (attempt 1): 1.0 * 2^1 = 2.0
    delay = exponential_backoff(1, config)
    assert delay == 2.0


def test_exponential_backoff_attempt_2():
    """Test exponential_backoff for third attempt."""
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

    # Third retry (attempt 2): 1.0 * 2^2 = 4.0
    delay = exponential_backoff(2, config)
    assert delay == 4.0


def test_exponential_backoff_max_delay():
    """Test exponential_backoff respects max_delay."""
    config = RetryConfig(
        base_delay=1.0,
        exponential_base=2.0,
        max_delay=10.0,
        jitter=False,
    )

    # Without max_delay would be 1.0 * 2^10 = 1024.0
    # Should be capped at 10.0
    delay = exponential_backoff(10, config)
    assert delay == 10.0


def test_exponential_backoff_with_jitter():
    """Test exponential_backoff includes jitter."""
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=True)

    # Run multiple times - jitter should cause variation
    delays = [exponential_backoff(1, config) for _ in range(10)]

    # Expected delay is 2.0, with jitter 1.0-3.0
    assert all(1.0 <= d <= 3.0 for d in delays)

    # Some variation expected (with jitter)
    assert len(set(delays)) > 1


def test_exponential_backoff_without_jitter():
    """Test exponential_backoff without jitter."""
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

    # Run multiple times - should be consistent
    delays = [exponential_backoff(1, config) for _ in range(10)]

    # Should all be exactly 2.0
    assert all(d == 2.0 for d in delays)


def test_exponential_backoff_custom_base():
    """Test exponential_backoff with custom exponential base."""
    config = RetryConfig(base_delay=1.0, exponential_base=3.0, jitter=False)

    # 1.0 * 3^2 = 9.0
    delay = exponential_backoff(2, config)
    assert delay == 9.0


def test_exponential_backoff_custom_base_delay():
    """Test exponential_backoff with custom base delay."""
    config = RetryConfig(base_delay=0.5, exponential_base=2.0, jitter=False)

    # 0.5 * 2^2 = 2.0
    delay = exponential_backoff(2, config)
    assert delay == 2.0


def test_retry_config_max_attempts_validation():
    """Test RetryConfig max_attempts must be positive."""
    # Valid
    config = RetryConfig(max_attempts=1)
    assert config.max_attempts == 1

    # Valid
    config = RetryConfig(max_attempts=100)
    assert config.max_attempts == 100


def test_retry_sequence():
    """Test a complete retry sequence."""
    config = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        exponential_base=2.0,
        jitter=False,
    )

    # Expected sequence:
    # Attempt 0 (first try): no delay
    # Attempt 1 (retry 1): 0.1 * 2^0 = 0.1s
    # Attempt 2 (retry 2): 0.1 * 2^1 = 0.2s
    # Total: 3 attempts

    delays = []
    for attempt in range(1, config.max_attempts):
        delay = exponential_backoff(attempt - 1, config)
        delays.append(delay)

    assert len(delays) == 2
    assert delays[0] == 0.1
    assert delays[1] == 0.2


def test_retry_backoff_progression():
    """Test that backoff increases exponentially."""
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

    previous_delay = 0
    for attempt in range(5):
        delay = exponential_backoff(attempt, config)
        assert delay >= previous_delay
        previous_delay = delay
