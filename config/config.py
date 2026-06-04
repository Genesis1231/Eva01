from pathlib import Path
import yaml
from pydantic import BaseModel, Field, AliasPath

CONFIG_FILE = Path(__file__).with_name("eva.yaml")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SQLITE_DB_NAME = "eva.db"

class Config(BaseModel):
    # System settings
    DEVICE: str = Field(validation_alias=AliasPath("system", "device"))
    LANGUAGE: str = Field(validation_alias=AliasPath("system", "language"))
    BASE_URL: str = Field(validation_alias=AliasPath("system", "base_url"))
    CAMERA: bool | int | str = Field(
        default=0,
        validation_alias=AliasPath("system", "camera")
    )

    HEARTBEAT_INTERVAL: int = Field(
        default=300,
        validation_alias=AliasPath("system", "heartbeat")
    )
    FEED_URL: str = Field(
        default="",
        validation_alias=AliasPath("system", "feed_url"),
    )

    # Model settings
    MAIN_MODEL: str = Field(validation_alias=AliasPath("models", "main"))
    VISION_MODEL: str = Field(validation_alias=AliasPath("models", "vision"))
    STT_MODEL: str = Field(validation_alias=AliasPath("models", "stt"))
    TTS_MODEL: str = Field(validation_alias=AliasPath("models", "tts"))
    UTILITY_MODEL: str = Field(validation_alias=AliasPath("models", "utility"))
    EMBEDDING_MODEL: str = Field(
        default="qwenvl:Qwen3-VL-Embedding-8B",
        validation_alias=AliasPath("models", "embedding"),
    )
    EMBEDDING_URL: str = Field(
        default="http://192.168.3.46:8000",
        validation_alias=AliasPath("models", "embedding_url"),
    )

    @classmethod
    def load_yaml(cls, path: Path = CONFIG_FILE) -> "Config":
        if not path.is_file():
            raise FileNotFoundError(
                f"EVA config file not found at '{path}'. "
                "Create config/eva.yaml before starting EVA."
            )
            
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return cls.model_validate(data)

# Config instance
eva_configuration = Config.load_yaml()
