"""
Configuration management for AI Video Analytics System.
"""
from typing import Optional, List, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    
    # Application Settings
    app_name: str = Field(default="AI Video Analytics System")
    log_level: str = Field(default="INFO")
    device: str = Field(default="cuda")
    
    # Model Configuration
    model_type: str = Field(default="yolov8")
    model_path: str = Field(default="models/yolov8n.pt")
    confidence_threshold: float = Field(default=0.5)
    iou_threshold: float = Field(default=0.45)
    
    # GPU Settings
    use_tensorrt: bool = Field(default=False)
    tensorrt_precision: str = Field(default="fp16")
    
    # Video Stream Configuration
    max_streams: int = Field(default=10)
    frame_skip: int = Field(default=0)
    resize_width: int = Field(default=640)
    resize_height: int = Field(default=480)
    
    # Alert Configuration
    alert_cooldown_seconds: int = Field(default=30)
    enable_webhook: bool = Field(default=False)
    webhook_url: Optional[str] = Field(default=None)
    enable_email: bool = Field(default=False)
    email_smtp_host: Optional[str] = Field(default=None)
    email_smtp_port: int = Field(default=587)
    email_from: Optional[str] = Field(default=None)
    email_to: Optional[str] = Field(default=None)
    
    # Database
    db_type: str = Field(default="sqlite")
    db_path: str = Field(default="alerts/alerts.db")
    db_host: Optional[str] = Field(default=None)
    db_port: int = Field(default=5432)
    db_name: Optional[str] = Field(default=None)
    db_user: Optional[str] = Field(default=None)
    db_password: Optional[str] = Field(default=None)
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=8000)


# Global settings instance
settings = Settings()
