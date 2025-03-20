from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class AppSettings(BaseSettings):
    AWS_REGION_NAME: str = "us-east-2"
    AWS_COGNITO_USER_POOL_ID: str = "us-east-2_onzgxu6oQ"
    AWS_COGNITO_APP_CLIENT_ID: str = "1t97t3qrackdsrrbi2tgh8qggm"

    model_config = SettingsConfigDict(env_file=".env")

settings = AppSettings()

@lru_cache
def get_settings():
    return settings

env_vars = get_settings() 