from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class AppSettings(BaseSettings):
    AWS_REGION_NAME: str
    AWS_COGNITO_USER_POOL_ID: str
    AWS_COGNITO_APP_CLIENT_ID: str

    model_config = SettingsConfigDict(env_file=".env")

# settings instance is created but might not be needed if get_settings is always used
settings = AppSettings()

@lru_cache
def get_settings():
    # Ensure a fresh instance is created potentially reading updated env vars/file
    return AppSettings()

# env_vars gets the settings instance at import time
env_vars = get_settings() 