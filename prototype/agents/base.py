"""
Base agent class defining the interface for all agents in the system.
All agents inherit from this and implement the run() method.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any
from prototype.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.

    Each agent is responsible for:
    1. Accepting the shared workflow state
    2. Performing its specific task
    3. Updating and returning the modified state
    4. Logging its actions and decisions

    Example:
        class MyAgent(BaseAgent):
            name = "MyAgent"

            def run(self, state: WorkflowState) -> WorkflowState:
                logger.info(f"{self.name} processing query: {state['user_query']}")
                # Do something
                state["my_result"] = result
                return state
    """

    # Subclasses must set this
    name: str = "BaseAgent"

    def __init__(self):
        """Initialize the agent with logging."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.name}")

    @abstractmethod
    def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the agent's task.

        This is the core method that all subclasses must implement.
        Each agent receives the current state, modifies it, and returns it.

        Args:
            state: Current workflow state

        Returns:
            Modified workflow state (can be the same object, modified in-place)

        Raises:
            Exception: If the agent encounters an error, it should be caught
                      and state['error_message'] should be set
        """
        pass

    def _log_step(self, step: str, details: Dict[str, Any] = None) -> None:
        """
        Log an agent's step for debugging.

        Args:
            step: Description of what the agent is doing
            details: Optional dict of details to log
        """
        if details:
            self.logger.debug(f"{self.name}: {step} - {details}")
        else:
            self.logger.debug(f"{self.name}: {step}")

    def _handle_error(self, state: WorkflowState, error: Exception) -> WorkflowState:
        """
        Standardized error handling for all agents.

        Args:
            state: Current workflow state
            error: Exception that occurred

        Returns:
            State with error message set
        """
        error_msg = f"{self.name} error: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        state["error_message"] = error_msg
        state["should_continue"] = False
        return state
