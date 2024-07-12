from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

# Define Model
llm_model = ChatOpenAI(model="gpt-3.5-turbo",
                temperature=0.1,
                api_key='fk219366-q0OeM3AXfZquVVy1dPUOqx4fc0QdQjJf',
                base_url="https://openai.api2d.net")



# Read Transcript
file = open("./chinese.txt", "r")
chinese_caption = file.read()
file.close()

file = open("./english.txt", "r")
english_caption = file.read()
file.close()

# Prompt Template
context_translation_template = """
native_subtitle = (
{chinese_caption}
)

foreign_subtitle = (
{english_caption}
)

----

native_subtitle is a subtitle translated from the foreign language subtitle, foreign_titles. 

Please find the time code of each occurrence of the following word in foreign_subtitle and the corresponding translation of the following word 
(not the entire sentence) that appears in the context of native_subtitle:

{word}

"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", context_translation_template, 
        ),
    ]
)



# Tool Calling (json output)

class ContextTranslation(BaseModel):
    """Get the current weather in a given location"""

    timecode: str = Field(..., description="the time code of the word that appears in foreign_subtitles")
    foreign: str = Field(..., description="the word in the foreign language")
    native: str = Field(..., description="the word in the native language")


llm_with_tools = llm_model.bind_tools([ContextTranslation])

# LLM Chain
chain = prompt | llm_with_tools

chain.invoke(
    {
        "chinese_caption" : chinese_caption,
        "english_caption" : english_caption, 
        "word" : "impression"
    }
)

