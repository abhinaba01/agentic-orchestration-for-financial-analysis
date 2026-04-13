from dataclasses import dataclass
from dotenv import load_dotenv
import os


@dataclass
class Config:
    """Configuration values loaded from environment variables."""

    OPENAI_API_KEY: str
    CHROMA_PERSIST_DIR: str
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    ONNX_NER_MODEL_PATH: str = "./models/ner_int8.onnx"
    ONNX_SENTIMENT_MODEL_PATH: str = "./models/sentiment_int8.onnx"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    GPT_MODEL: str = "gpt-4o"

    def validate(self) -> None:
        """Validate required configuration values and raise an error if invalid."""
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required and must be set in the environment."
            )


def get_config() -> Config:
    """Load environment variables and return a validated Config instance."""
    load_dotenv()
    config = Config(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
        CHROMA_PERSIST_DIR=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", "512")),
        CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", "50")),
        ONNX_NER_MODEL_PATH=os.getenv(
            "ONNX_NER_MODEL_PATH", "./models/ner_int8.onnx"
        ),
        ONNX_SENTIMENT_MODEL_PATH=os.getenv(
            "ONNX_SENTIMENT_MODEL_PATH", "./models/sentiment_int8.onnx"
        ),
        EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        GPT_MODEL=os.getenv("GPT_MODEL", "gpt-4o"),
    )
    config.validate()
    return config
