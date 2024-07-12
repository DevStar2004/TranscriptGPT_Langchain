import os
from dotenv import load_dotenv
import sqlite3
import numpy as np
import json
import pickle

from langchain_openai import OpenAI, OpenAIEmbeddings

# Load environment variables from the .env file
load_dotenv()

# Setup database and table
conn = sqlite3.connect('transcripts.db')
cursor = conn.cursor()

# Create a table to store the transcript data
cursor.execute('''
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    start REAL,
    end REAL,
    "foreign" TEXT,
    native TEXT,
    embedding BLOB
)
''')

# Commit the changes
conn.commit()

# Define the Embedding model
model = OpenAIEmbeddings(model="text-embedding-3-large",
                         api_key=os.getenv('OPENAI_API_KEY'),
                         base_url="https://openai.api2d.net")

with open('3xe5dY87syg_english.json', 'r', encoding='utf-8') as englishFile:
    with open('3xe5dY87syg_chinese.json', encoding='utf-8') as chineseFile:
        englishData = json.load(englishFile)
        chineseData = json.load(chineseFile)
        
        for i in range(0, len(englishData)):
            embedding = model.embed_query(englishData[i]['transcript'])
            embedding_blob = pickle.dumps(embedding)
            cursor.execute('INSERT INTO transcripts (id, start, end, "foreign", native, embedding) VALUES (?, ?, ?, ?, ?, ?)', 
                  (i + 1, englishData[i]['start'], englishData[i]['end'], englishData[i]['transcript'], 
                   chineseData['translated_segments'][i]['transcript'], sqlite3.Binary(embedding_blob)))
            print('Segment_' + str(i  + 1) + ' inserted into database')
        conn.commit()

conn.close()
# response  = model.embed_query('What is React.js?')
# print(len(response))
