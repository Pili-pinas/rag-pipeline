
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv('PINECONE_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
pc.create_index(
    name='pili-pinas-rag',
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1'),
)
print('Index created!')
