from pathlib import Path
import yaml
from pydantic import BaseModel, Field, AliasPath

CONFIG_FILE = Path(__file__).with_name("eva.yaml")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

class Config(BaseModel):
    # System settings
    DEVICE: str = Field(validation_alias=AliasPath("system", "device"))
    LANGUAGE: str = Field(validation_alias=AliasPath("system", "language"))
    BASE_URL: str = Field(validation_alias=AliasPath("system", "base_url"))
    CAMERA_URL: int | str = Field(
        default=0, 
        validation_alias=AliasPath("system", "camera_url")
    )

    # Model settings
    CHAT_MODEL: str = Field(validation_alias=AliasPath("models", "chat"))
    VISION_MODEL: str = Field(validation_alias=AliasPath("models", "vision"))
    STT_MODEL: str = Field(validation_alias=AliasPath("models", "stt"))
    TTS_MODEL: str = Field(validation_alias=AliasPath("models", "tts"))
    UTILITY_MODEL: str = Field(validation_alias=AliasPath("models", "utility"))

    @classmethod
    def load_yaml(cls, path: Path = CONFIG_FILE) -> "Config":
        if not path.is_file():
            raise FileNotFoundError(
                f"EVA config file not found at '{path}'. "
                "Create backend/app/config/eva.yaml before starting EVA."
            )
            
        with path.open("r", encoding="utf-8") as f:
            # Load YAML and validate it against the model
            data = yaml.safe_load(f) or {}
            return cls.model_validate(data)

# Config instance
eva_configuration = Config.load_yaml()