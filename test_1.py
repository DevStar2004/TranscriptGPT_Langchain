from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import ChatMessage

# Define the LLM model
llm_model = ChatOpenAI(model="gpt-3.5-turbo",
                       temperature=0.1,
                       api_key='fk219366-q0OeM3AXfZquVVy1dPUOqx4fc0QdQjJf',
                       base_url="https://openai.api2d.net")

# Define a conversation (list of messages) with proper roles
conversation_history = [[
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is React.js")
]]

# Generate a response from the model
response = llm_model.generate(messages=conversation_history)

# Print the response
print(response)
