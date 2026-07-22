"""
AgentResearch Configuration
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # DeepSeek API
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    deepseek_max_tokens: int = 4096
    deepseek_temperature: float = 0.3

    # Neon PostgreSQL
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "")
    )

    # App Settings
    app_name: str = field(
        default_factory=lambda: os.getenv("APP_NAME", "AgentResearch")
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # Agent Settings
    max_search_results: int = 5
    max_retries: int = 3
    agent_timeout_seconds: int = 120

    @property
    def deepseek_headers(self) -> dict:
        """Return headers for DeepSeek API requests."""
        return {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
        }

    def validate(self) -> list[str]:
        """Validate required settings. Returns list of missing keys."""
        missing = []
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if not self.database_url:
            missing.append("DATABASE_URL")
        return missing


# Global settings instance
settings = Settings()
