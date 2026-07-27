from enum import Enum, StrEnum, auto
from attr import dataclass
from openai import OpenAI
import os 

class Provider(StrEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"

@dataclass(frozen=True)
class ProviderConfig:
    base_url: str | None
    api_key_env: str
    default_model: str

CONFIGS: dict[Provider, ProviderConfig] = {
    Provider.OPENAI: ProviderConfig(
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4.1-mini"
    ),
    Provider.GEMINI: ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GOOGLE_API_KEY",
        default_model="gemini-3.5-flash-lite",
    ),
    Provider.GROQ: ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
    ),
    Provider.OLLAMA: ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env="",
        default_model="llama3.2",
    ),
}

class Model:
    def __init__(self, provider: Provider = Provider.OPENAI, model: str | None = None) -> None:
        
        self.provider = provider
        config = CONFIGS[self.provider]
        api_key = os.getenv(config.api_key_env) or "unused"
        self.model = model or config.default_model

        self.client = OpenAI(
            base_url=config.base_url,
            api_key=api_key
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        normalised_response = self.normalise_response(response)

        return normalised_response or ""

    def normalise_response(self, response) -> str:
        match self.provider:
            case Provider.OLLAMA:
                return response.choices[0].message.content.strip()
            case Provider.GEMINI:
                # Gemini via the compat layer sometimes wraps markdown in fences
                text = response.choices[0].message.content
                return text.removeprefix("```markdown").removesuffix("```").strip()
            case _:
                return response.choices[0].message.content