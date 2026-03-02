import os
from unittest.mock import patch, MagicMock

try:
    from dotenv import load_dotenv
    # Load .env from project root
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
except ImportError:
    pass

from eva.agent.chatagent import ChatAgent


def test_chatagent_initialization():
    print("Testing ChatAgent initialization...")
    with patch('eva.agent.chatagent.init_chat_model') as mock_init:
        mock_model = MagicMock()
        mock_init.return_value = mock_model

        agent = ChatAgent(model_name="test-model", base_url="http://localhost", language="english")

        assert agent.model_name == "test-model"
        assert agent.language == "english"
        mock_init.assert_called_once()
        print("Initialization test passed!")
        return agent, mock_model


def test_chatagent_respond():
    print("Testing ChatAgent respond method...")
    with patch('eva.agent.chatagent.init_chat_model') as mock_init:
        mock_model = MagicMock()
        mock_init.return_value = mock_model

        mock_ai_message = MagicMock()
        mock_ai_message.content = "This is a test response."
        mock_ai_message.tool_calls = []

        mock_bound_model = MagicMock()
        mock_bound_model.invoke.return_value = mock_ai_message
        mock_model.bind_tools.return_value = mock_bound_model

        agent = ChatAgent(model_name="test-model", base_url="http://localhost", language="english")

        result = agent.respond(
            sense={"user_message": "Hello?"},
            history=[{"role": "user", "content": "Hello?"}]
        )

        assert "response" in result
        assert result["response"] == "This is a test response."
        print("Respond method test passed!")


def test_chatagent_openai_real():
    print("\n--- Testing ChatAgent with real OpenAI model ---")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping real OpenAI test: OPENAI_API_KEY not set.")
        return

    agent = ChatAgent(model_name="gpt-4o-mini")

    sense_data = {"user_message": "generate exactly: 'Apple Banana Cherry' with valid JSON format."}

    print("Invoking agent...")
    result = agent.respond(sense=sense_data, history=[])

    print("\nReal response result:")
    for key, value in result.items():
        print(f"{key.upper()}: {value}")

    assert "response" in result
    print("\nOpenAI real test passed!")


if __name__ == "__main__":
    import sys
    test_chatagent_initialization()
    test_chatagent_respond()
    test_chatagent_openai_real()
    print("\nAll ChatAgent tests passed successfully!")
