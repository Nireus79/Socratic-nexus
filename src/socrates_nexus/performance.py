"""Performance optimization tools for LLM inference."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelOptimizationConfig:
    """Configuration for model optimization."""

    enable_quantization: bool = False
    quantization_bits: int = 8
    enable_caching: bool = True
    cache_size_mb: int = 512
    enable_batching: bool = True
    batch_size: int = 32
    max_sequence_length: int = 2048
    use_float16: bool = False
    enable_pruning: bool = False
    pruning_ratio: float = 0.1
    enable_distillation: bool = False
    distillation_temperature: float = 3.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for model inference."""

    request_id: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_used_mb: float
    cache_hit: bool
    batch_size: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationResult:
    """Result of optimization."""

    optimization_type: str
    improvement_percent: float
    latency_reduction_ms: float
    memory_reduction_mb: float
    cost_reduction_percent: float


class InferenceOptimizer:
    """Optimizes LLM inference performance."""

    def __init__(self, config: Optional[ModelOptimizationConfig] = None):
        """Initialize optimizer."""
        self.config = config or ModelOptimizationConfig()
        self.metrics: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []
        self.logger = logging.getLogger(__name__)

    def optimize_batch(
        self,
        inputs: List[str],
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Optimize batch processing of multiple inputs.

        Args:
            inputs: List of input strings
            model_name: Name of model to use

        Returns:
            Optimization metadata
        """
        batch_size = min(len(inputs), self.config.batch_size)

        optimization = {
            "batch_size": batch_size,
            "input_count": len(inputs),
            "estimated_speedup": len(inputs) / batch_size,
            "memory_efficient": batch_size <= 8,
        }

        return optimization

    def optimize_sequence_length(
        self,
        text: str,
        model_max_length: int,
    ) -> str:
        """
        Truncate or optimize sequence length.

        Args:
            text: Input text
            model_max_length: Maximum sequence length

        Returns:
            Optimized text within length limit
        """
        if len(text) <= model_max_length:
            return text

        # Smart truncation: keep most important parts
        sentences = text.split(". ")
        result = ""

        for sentence in sentences:
            if len(result) + len(sentence) <= model_max_length:
                result += sentence + ". "
            else:
                break

        return result.strip()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.metrics:
            return {"total_requests": 0, "cache_hits": 0, "hit_rate": 0}

        total = len(self.metrics)
        hits = sum(1 for m in self.metrics if m.cache_hit)

        return {
            "total_requests": total,
            "cache_hits": hits,
            "hit_rate": hits / total if total > 0 else 0,
            "estimated_latency_saved_ms": sum(
                m.latency_ms * 0.7 for m in self.metrics if m.cache_hit
            ),
        }

    def record_metric(
        self,
        request_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        memory_used_mb: float,
        cache_hit: bool = False,
        batch_size: int = 1,
    ) -> None:
        """Record inference metrics."""
        total_tokens = input_tokens + output_tokens
        throughput = (total_tokens / latency_ms * 1000) if latency_ms > 0 else 0

        metric = PerformanceMetrics(
            request_id=request_id,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            throughput_tokens_per_sec=throughput,
            memory_used_mb=memory_used_mb,
            cache_hit=cache_hit,
            batch_size=batch_size,
        )

        self.metrics.append(metric)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.metrics:
            return {"status": "no_data"}

        latencies = [m.latency_ms for m in self.metrics]
        throughputs = [m.throughput_tokens_per_sec for m in self.metrics]
        memory_usage = [m.memory_used_mb for m in self.metrics]

        return {
            "total_requests": len(self.metrics),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "avg_throughput": sum(throughputs) / len(throughputs),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
            "cache_stats": self.get_cache_stats(),
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if not self.metrics:
            return recommendations

        # Check latency
        avg_latency = sum(m.latency_ms for m in self.metrics) / len(self.metrics)
        if avg_latency > 500:
            recommendations.append("Consider enabling quantization to reduce latency")

        # Check cache hit rate
        cache_stats = self.get_cache_stats()
        if cache_stats["hit_rate"] < 0.3 and self.config.enable_caching:
            recommendations.append("Increase cache size to improve hit rate")

        # Check memory
        avg_memory = sum(m.memory_used_mb for m in self.metrics) / len(self.metrics)
        if avg_memory > 4000:
            recommendations.append("Enable model quantization to reduce memory usage")

        return recommendations


class LatencyOptimizer:
    """Optimizes request latency."""

    def __init__(self):
        """Initialize latency optimizer."""
        self.latency_samples: List[float] = []
        self.logger = logging.getLogger(__name__)

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.latency_samples.append(latency_ms)

    def detect_bottleneck(self) -> Optional[str]:
        """Detect performance bottleneck."""
        if not self.latency_samples:
            return None

        avg = sum(self.latency_samples) / len(self.latency_samples)
        max_latency = max(self.latency_samples)

        if max_latency > avg * 2:
            return "Outliers detected: some requests significantly slower"

        if avg > 1000:
            return "Average latency is high: consider optimization"

        return None

    def suggest_optimization(self) -> Optional[str]:
        """Suggest optimization strategy."""
        bottleneck = self.detect_bottleneck()

        if bottleneck and "Outliers" in bottleneck:
            return "Enable request queuing and load balancing"
        elif bottleneck and "high" in bottleneck:
            return "Enable model quantization or use smaller model"

        return "Performance acceptable"


class CostOptimizer:
    """Optimizes inference costs."""

    def __init__(self, pricing_per_1m_tokens: float = 0.5):
        """
        Initialize cost optimizer.

        Args:
            pricing_per_1m_tokens: Cost per 1 million tokens
        """
        self.pricing_per_1m_tokens = pricing_per_1m_tokens
        self.total_tokens_processed: int = 0
        self.logger = logging.getLogger(__name__)

    def estimate_cost(self, num_tokens: int) -> float:
        """Estimate cost for processing tokens."""
        return (num_tokens / 1_000_000) * self.pricing_per_1m_tokens

    def record_tokens(self, num_tokens: int) -> None:
        """Record tokens processed."""
        self.total_tokens_processed += num_tokens

    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost analysis report."""
        total_cost = self.estimate_cost(self.total_tokens_processed)

        return {
            "total_tokens": self.total_tokens_processed,
            "total_cost_usd": total_cost,
            "cost_per_token": (
                (total_cost / self.total_tokens_processed * 1_000_000)
                if self.total_tokens_processed > 0
                else 0
            ),
            "pricing_per_1m": self.pricing_per_1m_tokens,
        }

    def suggest_cost_reduction(self) -> List[str]:
        """Suggest cost reduction strategies."""
        suggestions = []

        if self.total_tokens_processed > 100_000:
            suggestions.append("Enable caching to reduce redundant token processing")
            suggestions.append("Batch requests to improve throughput efficiency")

        if self.pricing_per_1m_tokens > 1.0:
            suggestions.append("Consider using cheaper models")

        return suggestions
