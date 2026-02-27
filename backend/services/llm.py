import os

from backend.services.env import load_local_env

load_local_env()


class HFInferenceLLMService:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("HF_INFERENCE_MODEL", "openai/gpt-oss-20b")
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(model=self.model, token=self.api_key)
        return self._client

    def generate(self, prompt: str, max_new_tokens: int = 700) -> str:
        client = self._get_client()

        # Prefer chat-style completion when the model supports it.
        if hasattr(client, "chat_completion"):
            resp = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
            )
            choices = getattr(resp, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None)
                if content:
                    return str(content).strip()

        # Fallback to text generation APIs.
        if hasattr(client, "text_generation"):
            text = client.text_generation(prompt, max_new_tokens=max_new_tokens)
            return str(text).strip()

        raise RuntimeError("No supported HF inference generation method available")


llm_service = HFInferenceLLMService()
