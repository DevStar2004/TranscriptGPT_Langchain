import os
from dotenv import load_dotenv
import sqlite3
import numpy as np
import json
import pickle

from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

# Load environment variables from the .env file
load_dotenv()

start_time = 0
end_time = 12
word = "restaurant"

# Get Embedding of Word
# Define the Embedding model
model = OpenAIEmbeddings(model="text-embedding-3-large",
                         api_key=os.getenv('OPENAI_API_KEY'),
                         base_url="https://openai.api2d.net")

word_embedding = model.embed_query(word)
# print(word_embedding)

# Setup database connection
conn = sqlite3.connect('transcripts.db')
cursor = conn.cursor()


# Fetch Segments between the timestamp
cursor.execute('''
SELECT * FROM transcripts
WHERE start >= ? AND end <= ?
''', (start_time, end_time))

segments = cursor.fetchall()

english_caption = ''
chinese_caption = ''

for segment in segments:
    sentence_embedding = pickle.loads(segment[5])
    sentence_embedding = np.array(sentence_embedding)
    # print(segment[3])
    similarity = cosine_similarity(sentence_embedding.reshape(1 ,-1), np.array(word_embedding).reshape(1, -1))
    
    if similarity >= 0.1:
        english_caption += '\n' + str(segment[1]) + ' -> ' + str(segment[2]) + '\n' + segment[3]
        chinese_caption += '\n' + str(segment[1]) + ' -> ' + str(segment[2]) + '\n' + segment[4]
        

llm_model = ChatOpenAI(model="gpt-3.5-turbo",
                temperature=0.1,
                base_url="https://openai.api2d.net")




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

class ContextTranslation(BaseModel):
    """Get the current weather in a given location"""

    timecode: str = Field(..., description="the time code of the word that appears in foreign_subtitles")
    foreign: str = Field(..., description="the word in the foreign language")
    native: str = Field(..., description="the word in the native language")


llm_with_tools = llm_model.bind_tools([ContextTranslation])
chain = prompt | llm_with_tools

output = chain.invoke(
    {
        "chinese_caption" : chinese_caption,
        "english_caption" : english_caption, 
        "word" : word
    }
)

print(output)