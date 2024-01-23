from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
# Get the Transcript from YT
video_url = "https://www.youtube.com/watch?v=o3K_HbpWNpgs"
try:
    video_id = video_url.split("v=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    if transcript:
        subtitles_text = " ".join([entry["text"] for entry in transcript])
        #print(subtitles_text)
    else:
        print("TS NA")
except Exception as e:
    print(f"An error occured:{str(e)}")

#QDRANT Create your client

from qdrant_client import QdrantClient
os.environ['QDRANT_HOST'] 
os.environ['QDRANT_API_KEY'] 

client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

# Create Collection
os.environ['QDRANT_COLLECTION'] =''

collection_config = qdrant_client.http.models.VectorParams(
        size=384, # 768 for instructor-xl, 1536 for OpenAI
        distance=qdrant_client.http.models.Distance.COSINE
    )

client.recreate_collection(
    collection_name=os.getenv("QDRANT_COLLECTION"),
    vectors_config=collection_config
)

# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
# )

# query_result = embeddings.embed_query(subtitles_text)
# print(query_result)

from langchain.text_splitter import CharacterTextSplitter

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
print(get_chunks(subtitles_text))