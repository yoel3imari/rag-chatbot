"""Conversion between stored message dictionaries and LangChain messages."""

from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def to_langchain_messages(raw_messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """Converts dict messages into LangChain BaseMessage instances."""
    messages: List[BaseMessage] = []
    for message in raw_messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages
