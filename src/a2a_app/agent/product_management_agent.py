import json
import logging
import os
from collections.abc import AsyncIterable
from typing import Any, Annotated, Literal

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from agent_framework import Agent, AgentSession, tool
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatOptions

logger = logging.getLogger(__name__)
load_dotenv()


# region Chat Service Configuration

def get_chat_client() -> FoundryChatClient:
    """
    Return Azure AI Foundry chat client using the project endpoint and AAD.
    """
    endpoint = os.getenv("gpt_endpoint")
    deployment_name = os.getenv("gpt_deployment")

    if not endpoint:
        raise ValueError("gpt_endpoint is required")
    if not deployment_name:
        raise ValueError("gpt_deployment is required")

    return FoundryChatClient(
        project_endpoint=endpoint.rstrip("/"),
        model=deployment_name,
        credential=DefaultAzureCredential(),
    )


# endregion

# region Product Tool

@tool
def get_products(
    question: Annotated[
        str,
        'Natural language query to retrieve products, e.g. "What kinds of paint rollers do you have in stock?"',
    ],
) -> list[dict[str, Any]]:
    """
    Mock product lookup tool.
    In a real implementation, this would query a database or service.
    """
    _ = question  # Reserved for future filtering / retrieval logic.

    product_dict = [
        {
            "id": "1",
            "name": "Eco-Friendly Paint Roller",
            "type": "Paint Roller",
            "description": "A high-quality, eco-friendly paint roller for smooth finishes.",
            "punchLine": "Roll with the best, paint with the rest!",
            "price": 15.99,
            "quantity": 24,
        },
        {
            "id": "2",
            "name": "Premium Paint Brush Set",
            "type": "Paint Brush",
            "description": "A set of premium paint brushes for detailed work and fine finishes.",
            "punchLine": "Brush up your skills with our premium set!",
            "price": 25.49,
            "quantity": 16,
        },
        {
            "id": "3",
            "name": "All-Purpose Paint Tray",
            "type": "Paint Tray",
            "description": "A durable paint tray suitable for all types of rollers and brushes.",
            "punchLine": "Tray it, paint it, love it!",
            "price": 9.99,
            "quantity": 31,
        },
        {
            "id": "4",
            "name": "Standard Paint Roller",
            "type": "Paint Roller",
            "description": "A reliable standard paint roller for everyday painting jobs.",
            "punchLine": "Simple, steady, and ready to roll.",
            "price": 11.99,
            "quantity": 42,
        },
    ]

    return product_dict


# endregion

# region Response Format

class ResponseFormat(BaseModel):
    """
    Structured response schema returned by the model.
    """

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


# endregion

# region Agent Framework Agent

class AgentFrameworkProductManagementAgent:
    """
    Wraps a Microsoft Agent Framework agent for Zava product management tasks.
    """

    agent: Agent
    session: AgentSession | None = None
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        chat_service = get_chat_client()

        # Marketing Agent
        marketing_agent = Agent(
            client=chat_service,
            name="MarketingAgent",
            instructions=(
                "You specialize in planning and recommending marketing strategies for products. "
                "This includes identifying target audiences, making product descriptions better, "
                "suggesting upselling or cross-selling ideas, and proposing promotional tactics. "
                "Your goal is to help businesses effectively market their products and reach their desired customers.\n\n"
                "Always return a valid JSON object with this exact structure:\n"
                '{"status":"completed","message":"..."}\n'
                'or {"status":"input_required","message":"..."}\n'
                'or {"status":"error","message":"..."}'
            ),
            tools=[],
        )

        # Ranker Agent
        ranker_agent = Agent(
            client=chat_service,
            name="RankerAgent",
            instructions=(
                "You specialize in ranking and recommending products based on criteria such as budget, "
                "use case, practicality, and product fit. "
                "Use only the information provided in the prompt or tool outputs. "
                "Do not invent product facts.\n\n"
                "Always return a valid JSON object with this exact structure:\n"
                '{"status":"completed","message":"..."}\n'
                'or {"status":"input_required","message":"..."}\n'
                'or {"status":"error","message":"..."}'
            ),
            tools=[],
        )

        # Product Agent
        product_agent = Agent(
            client=chat_service,
            name="ProductAgent",
            instructions=(
                "You specialize in handling product-related requests from customers and employees. "
                "This includes listing products, identifying available quantities, providing prices, "
                "and giving product descriptions exactly as they exist in the catalog.\n\n"
                "You MUST use the get_products tool to answer all product-related questions. "
                "You MUST NEVER answer from your own knowledge. "
                "You MUST ONLY use products from the get_products tool. "
                "Do not make up product information.\n\n"
                "Always return a valid JSON object with this exact structure:\n"
                '{"status":"completed","message":"..."}\n'
                'or {"status":"input_required","message":"..."}\n'
                'or {"status":"error","message":"..."}'
            ),
            tools=[get_products],
        )

        # Main Product Manager Agent
        self.agent = Agent(
            client=chat_service,
            name="ProductManagerAgent",
            instructions=(
                "Your role is to carefully analyze the user's request and respond as best as you can. "
                "Your primary goal is precise and efficient delegation to ensure customers and employees receive accurate and specialized assistance promptly.\n\n"
                "Routing rules:\n"
                "- Whenever a user query is related to retrieving product information, pricing, inventory, or catalog details, delegate to ProductAgent.\n"
                "- Use MarketingAgent for marketing-related queries, improving descriptions, promotional language, upselling, and positioning.\n"
                "- Use RankerAgent for ranking, comparison, recommendation, and best-choice questions.\n"
                "- You may use these agents in conjunction with each other to provide a complete answer.\n\n"
                "If the user asks to improve a product description, first get the relevant product details through ProductAgent when needed, then use MarketingAgent.\n"
                "If the user asks for a recommendation, get product facts through ProductAgent when needed, then use RankerAgent.\n\n"
                "Always return a valid JSON object with this exact structure:\n"
                '{"status":"completed","message":"..."}\n'
                'or {"status":"input_required","message":"..."}\n'
                'or {"status":"error","message":"..."}\n'
                "Do not return plain text outside the JSON object."
            ),
            tools=[
                product_agent.as_tool(),
                marketing_agent.as_tool(),
                ranker_agent.as_tool(),
            ],
        )

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        """
        Handle non-streaming requests.
        """
        await self._ensure_session_exists(session_id)

        response = await self.agent.run(
            messages=user_input,
            session=self.session,
            options=OpenAIChatOptions(response_format=ResponseFormat),
        )

        return self._get_agent_response(response.text)

    async def stream(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Handle streaming requests.
        """
        await self._ensure_session_exists(session_id)

        chunks: list[str] = []

        async for chunk in self.agent.run_stream(
            messages=user_input,
            session=self.session,
        ):
            text = getattr(chunk, "text", None)
            if text:
                chunks.append(str(text))

        if chunks:
            yield self._get_agent_response("".join(chunks))

    def _get_agent_response(self, message: Any) -> dict[str, Any]:
        """
        Parse the model response into the app's expected structure.
        """
        message_text = str(message)

        default_response = {
            "is_task_complete": True,
            "require_user_input": False,
            "content": message_text,
        }

        try:
            structured_response = ResponseFormat.model_validate_json(message_text)
        except ValidationError:
            logger.info("Message did not come in JSON format.")
            return default_response
        except Exception:
            logger.exception("Unexpected error while processing the message.")
            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "We are unable to process your request at the moment. Please try again.",
            }

        response_map = {
            "input_required": {
                "is_task_complete": False,
                "require_user_input": True,
            },
            "error": {
                "is_task_complete": False,
                "require_user_input": True,
            },
            "completed": {
                "is_task_complete": True,
                "require_user_input": False,
            },
        }

        response = response_map.get(structured_response.status)
        if response:
            return {**response, "content": structured_response.message}

        return default_response

    async def _ensure_session_exists(self, session_id: str) -> None:
        """
        Ensure a session exists for the provided session ID.
        """
        if self.session is None or self.session.service_session_id != session_id:
            self.session = self.agent.create_session(session_id=session_id)


# endregion