"""Tests for performance optimization module."""

import pytest
from socrates_nexus.performance import (
    PerformanceMetrics,
    InferenceOptimizer,
    LatencyOptimizer,
    CostOptimizer,
    ModelOptimizationConfig,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""

    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            input_tokens=100,
            output_tokens=50,
            latency_ms=150.5,
            cost_usd=0.05,
        )
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.latency_ms == 150.5
        assert metrics.cost_usd == 0.05

    def test_metrics_default_values(self):
        """Test metrics with default values."""
        metrics = PerformanceMetrics()
        assert metrics.input_tokens is not None or metrics.input_tokens == 0
        assert metrics.output_tokens is not None or metrics.output_tokens == 0


class TestModelOptimizationConfig:
    """Test ModelOptimizationConfig class."""

    def test_config_creation(self):
        """Test creating optimization config."""
        config = ModelOptimizationConfig(
            model="gpt-4",
            target_latency_ms=500,
            budget_usd=1.0,
        )
        assert config.model == "gpt-4"
        assert config.target_latency_ms == 500
        assert config.budget_usd == 1.0

    def test_config_defaults(self):
        """Test config with default values."""
        config = ModelOptimizationConfig(model="gpt-4")
        assert config.model == "gpt-4"


class TestInferenceOptimizer:
    """Test InferenceOptimizer class."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = ModelOptimizationConfig(model="gpt-4")
        optimizer = InferenceOptimizer(config)

        assert optimizer.config.model == "gpt-4"

    def test_optimizer_methods_exist(self):
        """Test that optimizer has required methods."""
        config = ModelOptimizationConfig(model="gpt-4")
        optimizer = InferenceOptimizer(config)

        # Check that key methods exist
        assert hasattr(optimizer, "optimize")
        assert hasattr(optimizer, "get_recommendations")


class TestLatencyOptimizer:
    """Test LatencyOptimizer class."""

    def test_latency_optimizer_init(self):
        """Test latency optimizer initialization."""
        config = ModelOptimizationConfig(model="gpt-4", target_latency_ms=500)
        optimizer = LatencyOptimizer(config)

        assert optimizer.config.target_latency_ms == 500

    def test_latency_optimizer_methods(self):
        """Test that latency optimizer has required methods."""
        config = ModelOptimizationConfig(model="gpt-4")
        optimizer = LatencyOptimizer(config)

        assert hasattr(optimizer, "optimize")


class TestCostOptimizer:
    """Test CostOptimizer class."""

    def test_cost_optimizer_init(self):
        """Test cost optimizer initialization."""
        config = ModelOptimizationConfig(model="gpt-4", budget_usd=1.0)
        optimizer = CostOptimizer(config)

        assert optimizer.config.budget_usd == 1.0

    def test_cost_optimizer_methods(self):
        """Test that cost optimizer has required methods."""
        config = ModelOptimizationConfig(model="gpt-4")
        optimizer = CostOptimizer(config)

        assert hasattr(optimizer, "optimize")


class TestOptimizationIntegration:
    """Integration tests for optimization."""

    def test_create_and_use_optimizer(self):
        """Test creating and using optimizer."""
        config = ModelOptimizationConfig(
            model="gpt-4",
            target_latency_ms=1000,
            budget_usd=10.0,
        )
        optimizer = InferenceOptimizer(config)

        assert optimizer is not None
        assert optimizer.config == config

    def test_multiple_optimizers(self):
        """Test using multiple optimizer types."""
        config = ModelOptimizationConfig(model="gpt-4")

        latency_opt = LatencyOptimizer(config)
        cost_opt = CostOptimizer(config)
        general_opt = InferenceOptimizer(config)

        assert latency_opt is not None
        assert cost_opt is not None
        assert general_opt is not None
