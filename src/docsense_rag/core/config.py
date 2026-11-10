from pydantic_settings import BaseSettings, SettingsConfigDict

"""
BaseSettings - permite acessar variáveis de ambiente e validá-las
SettingsConfigDict - classe que define o comportamento de carregamento  de variáveis de ambiente, segredos e diretórios
"""


class Settings(BaseSettings):
    app_name: str = "DocSense RAG"
    environment: str = "dev"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DOCSENSE_",
        extra="ignore",
    )


settings = Settings()
