"""
Abstract base class for chat-completion API clients.
Used by the NamingService to generate cluster names and pain descriptions via LLM.
"""

from abc import ABC, abstractmethod


class BaseCompletionClient(ABC):
    """
    Abstraction over any chat-completion API (e.g. OpenAI, Anthropic, Bedrock).

    The concrete subclass is responsible for:
    - API URL and authentication
    - Model selection and parameters (temperature, max_tokens, …)
    - Request/response serialisation
    - Rate limit handling and retries

    This base class defines only the contract that the NamingService depends on.
    """

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response as a plain string.

        Args:
            system_prompt: Instruction context for the model.
            user_prompt:   User-turn message.

        Returns:
            The model's text response, stripped of leading/trailing whitespace.

        Raises:
            Should raise on unrecoverable errors after internal retries.
        """

    @abstractmethod
    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Send a prompt expecting a JSON response and return the parsed dict.

        Implementations must:
        1. Call the LLM (with JSON-mode / structured output if available).
        2. Parse the response as JSON.
        3. Validate the result is a non-empty dict.
        4. Raise if parsing fails after retries.

        Args:
            system_prompt: Instruction context, should specify JSON format.
            user_prompt:   User-turn message.

        Returns:
            Parsed JSON as a Python dict.

        Raises:
            Should raise on JSON parse failures or invalid shapes.
        """
