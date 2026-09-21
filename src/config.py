from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    openai_api_key: str = "changeme"
    openai_base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-chat"
    temperature: float = 0.7

    # RAGFlow
    ragflow_base_url: str = "http://192.168.138.137:81"
    ragflow_api_key: str = "changeme"
    ragflow_kb_id: str = "changeme"

    # Amap
    amap_api_key: str = "changeme"

    # Tavily
    tavily_api_key: str = "changeme"

    # Pexels
    pexels_api_key: str = "changeme"

    # Tencent COS
    cos_secret_id: str = "changeme"
    cos_secret_key: str = "changeme"
    cos_region: str = "ap-guangzhou"
    cos_bucket_name: str = "changeme"

    # Server
    server_port: int = 8123
    server_host: str = "0.0.0.0"


settings = Settings()
