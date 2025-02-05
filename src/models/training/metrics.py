from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from collections import defaultdict

@dataclass
class MetricsTracker:
    """Tracks training metrics over time."""
    metrics: Dict[str, list] = field(default_factory=lambda: defaultdict(list))
    latest_metrics: Dict[str, float] = field(default_factory=dict)

    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.latest_metrics[key] = value

    def get_metric_history(self, metric_name: str) -> list:
        """Get history of a specific metric."""
        return self.metrics.get(metric_name, [])

    def get_latest_metrics(self) -> Dict[str, float]:
        """Get most recent values for all metrics."""
        return self.latest_metrics.copy()

    def get_average(self, metric_name: str) -> Optional[float]:
        """Calculate average for a metric."""
        values = self.metrics.get(metric_name)
        if values:
            return sum(values) / len(values)
        return None
