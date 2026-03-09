"""Retry logic with exponential backoff."""

import time
import random
from typing import Callable, Any, Optional
from functools import wraps

from .exceptions import RateLimitError


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 32.0,
    jitter: bool = True,
):
    """
    Decorator for retrying with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter to delay
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, TimeoutError, ConnectionError) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        # Add jitter if enabled
                        if jitter:
                            delay_with_jitter = delay * (0.5 + random.random())
                        else:
                            delay_with_jitter = delay

                        # Cap delay at max_delay
                        delay_with_jitter = min(delay_with_jitter, max_delay)

                        # Extract retry_after if available
                        if isinstance(e, RateLimitError) and e.retry_after:
                            delay_with_jitter = max(delay_with_jitter, e.retry_after)

                        time.sleep(delay_with_jitter)
                        delay = min(delay * backoff_factor, max_delay)

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def exponential_backoff(
    attempt: int,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 32.0,
    jitter: bool = True,
) -> float:
    """
    Calculate exponential backoff delay.

    Args:
        attempt: Attempt number (0-indexed)
        backoff_factor: Multiplier for each retry
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    delay = initial_delay * (backoff_factor ** attempt)
    delay = min(delay, max_delay)

    if jitter:
        delay = delay * (0.5 + random.random())

    return delay
