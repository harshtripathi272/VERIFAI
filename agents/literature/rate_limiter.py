"""
Smart Rate Limiter for Literature API Calls

Handles multiple API keys with different rate limits intelligently.
Automatically selects the best available key and implements token bucket algorithm.
"""

import time
import threading
from collections import deque
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class APIProvider(Enum):
    """API providers with their respective rate limits."""
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    EUROPE_PMC = "europe_pmc"


@dataclass
class APIKey:
    """Represents an API key with its rate limit configuration."""
    key: str
    provider: APIProvider
    requests_per_second: float
    max_burst: int = 10  # Maximum burst requests
    
    def __post_init__(self):
        self.request_times: deque = deque(maxlen=self.max_burst)
        self.lock = threading.Lock()


class RateLimiter:
    """
    Intelligent rate limiter with multiple key support.
    
    Features:
    - Automatic key rotation
    - Token bucket algorithm
    - Per-key rate tracking
    - Optimal key selection
    """
    
    def __init__(self):
        self.api_keys: Dict[APIProvider, List[APIKey]] = {
            APIProvider.PUBMED: [],
            APIProvider.SEMANTIC_SCHOLAR: [],
            APIProvider.EUROPE_PMC: []
        }
        self.global_lock = threading.Lock()
    
    def add_key(self, provider: APIProvider, key: str, requests_per_second: float, max_burst: int = 10):
        """Add an API key to the rate limiter."""
        api_key = APIKey(
            key=key,
            provider=provider,
            requests_per_second=requests_per_second,
            max_burst=max_burst
        )
        self.api_keys[provider].append(api_key)
        print(f"[RateLimiter] Added {provider.value} key: {requests_per_second} req/s")
    
    def _get_best_key(self, provider: APIProvider) -> Optional[APIKey]:
        """
        Select the best available key based on current usage.
        Prefers keys with higher rate limits that have capacity.
        """
        keys = self.api_keys.get(provider, [])
        if not keys:
            return None
        
        # Sort by requests_per_second (descending) and pick the first available
        sorted_keys = sorted(keys, key=lambda k: k.requests_per_second, reverse=True)
        
        for key in sorted_keys:
            if self._has_capacity(key):
                return key
        
        # If no key has immediate capacity, return the fastest one (will wait)
        return sorted_keys[0]
    
    def _has_capacity(self, key: APIKey) -> bool:
        """Check if a key has capacity for immediate request."""
        with key.lock:
            current_time = time.time()
            
            # Remove old requests outside the time window
            window = 1.0 / key.requests_per_second
            while key.request_times and current_time - key.request_times[0] > window:
                key.request_times.popleft()
            
            # Check if we can make a request now
            return len(key.request_times) < key.max_burst
    
    def _wait_for_capacity(self, key: APIKey):
        """Wait until the key has capacity for a request."""
        with key.lock:
            while True:
                current_time = time.time()
                window = 1.0 / key.requests_per_second
                
                # Remove old requests
                while key.request_times and current_time - key.request_times[0] > window:
                    key.request_times.popleft()
                
                # Check capacity
                if len(key.request_times) < key.max_burst:
                    key.request_times.append(current_time)
                    return
                
                # Calculate wait time
                if key.request_times:
                    oldest_request = key.request_times[0]
                    wait_time = window - (current_time - oldest_request)
                    if wait_time > 0:
                        time.sleep(wait_time + 0.01)  # Small buffer
    
    def execute_with_rate_limit(
        self,
        provider: APIProvider,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with rate limiting.
        
        Args:
            provider: API provider
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function call
        """
        key = self._get_best_key(provider)
        
        if key is None:
            # No API key configured, execute without rate limiting
            return func(*args, **kwargs)
        
        # Wait for capacity
        self._wait_for_capacity(key)
        
        # Execute the function with the selected key
        # Note: The function should accept 'api_key' parameter if needed
        try:
            if 'api_key' in func.__code__.co_varnames:
                return func(*args, api_key=key.key, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            print(f"[RateLimiter] Error with {provider.value}: {e}")
            raise


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def configure_rate_limiter_from_settings():
    """Configure rate limiter from app settings."""
    from app.config import settings
    
    limiter = get_rate_limiter()
    
    # Configure PubMed/NCBI keys
    if hasattr(settings, 'NCBI_API_KEYS') and settings.NCBI_API_KEYS:
        # Multiple keys with different limits
        for key_config in settings.NCBI_API_KEYS:
            limiter.add_key(
                APIProvider.PUBMED,
                key_config['key'],
                key_config.get('requests_per_second', 10),
                key_config.get('max_burst', 10)
            )
    elif settings.NCBI_API_KEY:
        # Single key (default: 10 req/s with key, 3 req/s without)
        limiter.add_key(APIProvider.PUBMED, settings.NCBI_API_KEY, 10, 10)
    
    # Configure Semantic Scholar keys
    if hasattr(settings, 'SEMANTIC_SCHOLAR_API_KEYS') and settings.SEMANTIC_SCHOLAR_API_KEYS:
        for key_config in settings.SEMANTIC_SCHOLAR_API_KEYS:
            limiter.add_key(
                APIProvider.SEMANTIC_SCHOLAR,
                key_config['key'],
                key_config.get('requests_per_second', 1),
                key_config.get('max_burst', 5)
            )
    elif settings.SEMANTIC_SCHOLAR_API_KEY:
        limiter.add_key(APIProvider.SEMANTIC_SCHOLAR, settings.SEMANTIC_SCHOLAR_API_KEY, 1, 5)
    
    print("[RateLimiter] Configuration complete")
    return limiter
