import os
from datetime import datetime


def create_timestamped_output_dir(base_dir: str) -> str:
    """Create a timestamped directory for outputs.

    Args:
        base_dir: Base directory for outputs

    Returns:
        Created directory path with timestamp
    """
    # Create timestamp string in format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    return output_dir