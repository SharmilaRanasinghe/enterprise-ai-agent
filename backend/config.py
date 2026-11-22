
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_api_key: str
    gemini_model: str = "gemini-3.6-flash"

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "enterprise-ai-agent"

    pinecone_api_key: str = ""
    pinecone_index_name: str = "enterprise-knowledge"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
