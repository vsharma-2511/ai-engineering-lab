import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Default Provider Selection
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    llm_provider: str = LLM_PROVIDER  # Alias for lowercase access

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")

    # Aliases for lowercase access
    gemini_api_key: str = GEMINI_API_KEY
    openai_api_key: str = OPENAI_API_KEY
    anthropic_api_key: str = ANTHROPIC_API_KEY
    serpapi_api_key: str = SERPAPI_API_KEY

    # Default Models per Provider
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    # Aliases for lowercase access
    gemini_model: str = GEMINI_MODEL
    openai_model: str = OPENAI_MODEL
    anthropic_model: str = ANTHROPIC_MODEL

    @classmethod
    def get_default_model(cls, provider: str) -> str:
        models = {
            "gemini": cls.GEMINI_MODEL,
            "openai": cls.OPENAI_MODEL,
            "anthropic": cls.ANTHROPIC_MODEL,
        }
        return models.get(provider.lower(), cls.GEMINI_MODEL)

    @classmethod
    def get_api_key(cls, provider: str) -> str:
        keys = {
            "gemini": cls.GEMINI_API_KEY,
            "openai": cls.OPENAI_API_KEY,
            "anthropic": cls.ANTHROPIC_API_KEY,
            "serpapi": cls.SERPAPI_API_KEY,
        }
        key = keys.get(provider.lower(), "")
        if not key:
            raise ValueError(
                f"Missing API key for provider '{provider}'. "
                f"Please set the corresponding environment variable in your .env file."
            )
        return key


settings = Settings()