from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_path: Path = Path("saved_models/resnet18_cifar10.pt")
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 8000


settings = Settings()
