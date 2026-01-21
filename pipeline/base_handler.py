from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd


class BaseHandler(ABC):
    """
    Abstract base class for all data processing handlers

    Implements the Chain of Responsibility pattern where each handler
    processes data and passes it to the next handler in the chain
    """

    def __init__(self) -> None:
        """Initialize the handler with no successor"""
        self._next_handler: Optional['BaseHandler'] = None

    def set_next(self, handler: 'BaseHandler') -> 'BaseHandler':
        """
        Set the next handler in the chain

        Args:
            handler: The next handler to process data

        Returns:
            The handler that was set, allowing for method chaining
        """
        self._next_handler = handler
        return handler

    def handle(self, data: Any) -> Any:
        """
        Process data and pass it to the next handler

        Args:
            data: Data to be processed (typically a pandas DataFrame)

        Returns:
            Processed data from the entire chain
        """
        # Process data in current handler
        processed_data = self._process(data)

        # Pass to next handler if exists
        if self._next_handler:
            return self._next_handler.handle(processed_data)

        return processed_data

    @abstractmethod
    def _process(self, data: Any) -> Any:
        """
        Abstract method to be implemented by concrete handlers.

        Args:
            data: Data to be processed

        Returns:
            Processed data
        """
        pass
