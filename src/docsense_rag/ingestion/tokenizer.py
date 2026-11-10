from typing import Protocol


class Tokenizer(Protocol):
    """Interface para codificação e decodificação de tokens"""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...
