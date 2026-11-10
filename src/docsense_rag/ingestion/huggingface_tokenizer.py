from transformers import AutoTokenizer


class HuggingFaceTokenizer:
    """Tokeniza textos usando o tokenizer correspondente ao modelo."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> list[int]:
        """Codifica sem adicionar tokens especiais ao contar o conteúdo."""
        return self._tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def decode(self, token_ids: list[int]) -> str:
        text = self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if not isinstance(text, str):
            raise TypeError(
                "O tokenizer deveria retornar uma string ao decodificar tokens."
            )

        return text
