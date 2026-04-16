from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # This will look for an env var named OPENAI_API_KEY
    openai_api_key: str 
    
    # Tells Pydantic to read from .env if it exists
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()