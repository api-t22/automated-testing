from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    # Default fallback
    model: str = Field("gpt-4o-mini", env="MODEL")

    # Stratified Models
    model_simple: str = Field("gpt-5-nano-2025-08-07", env="MODEL_SIMPLE")
    model_complex: str = Field("gpt-5-mini-2025-08-07", env="MODEL_COMPLEX")

    max_retries: int = 2
