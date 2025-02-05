"""AWS EC2 instance management utilities."""

import boto3
from botocore.exceptions import ClientError
import requests
from typing import Optional
import time
from pathlib import Path
import json
from dataclasses import dataclass
from src.utils.project_logger import ProjectLogger

@dataclass
class InstanceInfo:
    """Information about the current EC2 instance."""
    instance_id: str
    region: str
    instance_type: str

class InstanceManager:
    """Manages EC2 instance lifecycle."""
    
    def __init__(self, logger: ProjectLogger):
        self.logger = logger
        self.instance_info = self._get_instance_info()
        self.ec2_client = boto3.client('ec2', region_name=self.instance_info.region)

    def _get_instance_info(self) -> InstanceInfo:
        """Get current instance metadata."""
        try:
            # Get instance metadata
            response = requests.get(
                "http://169.254.169.254/latest/dynamic/instance-identity/document",
                timeout=2
            )
            metadata = response.json()
            
            return InstanceInfo(
                instance_id=metadata['instanceId'],
                region=metadata['region'],
                instance_type=metadata['instanceType']
            )
        except Exception as e:
            self.logger.error(f"Failed to get instance metadata: {e}")
            raise

    def stop_instance(self) -> None:
        """Stop the current EC2 instance."""
        try:
            self.logger.info(
                "Stopping instance",
                instance_id=self.instance_info.instance_id
            )
            
            self.ec2_client.stop_instances(
                InstanceIds=[self.instance_info.instance_id]
            )
            
            self.logger.info("Instance stop command sent successfully")
            
        except ClientError as e:
            self.logger.error(f"Failed to stop instance: {e}")
            raise

    def terminate_instance(self) -> None:
        """Terminate the current EC2 instance."""
        try:
            self.logger.info(
                "Terminating instance",
                instance_id=self.instance_info.instance_id
            )
            
            self.ec2_client.terminate_instances(
                InstanceIds=[self.instance_info.instance_id]
            )
            
            self.logger.info("Instance termination command sent successfully")
            
        except ClientError as e:
            self.logger.error(f"Failed to terminate instance: {e}")
            raise
