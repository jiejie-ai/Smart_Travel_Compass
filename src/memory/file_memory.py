import pickle
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class FileBasedChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, base_dir: str, conversation_id: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        safe_id = conversation_id.replace("/", "_").replace("\\", "_")
        self.file_path = self.base_dir / f"{safe_id}.pickle"
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        self._ensure_loaded()
        return self._messages

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self._ensure_loaded()
        self._messages.extend(messages)
        self._save()

    def clear(self) -> None:
        self._messages = []
        if self.file_path.exists():
            self.file_path.unlink()

    def _ensure_loaded(self):
        if not self._messages and self.file_path.exists():
            with open(self.file_path, "rb") as f:
                self._messages = pickle.load(f)

    def _save(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self._messages, f)
