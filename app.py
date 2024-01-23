from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

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

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(subtitles_text)
# print(embeddings)

#QDRANT Create your client

from qdrant_client import QdrantClient

# qdrant_client = QdrantClient(
#     url="https://3036137f-6be5-4cd3-90da-38d4e0f8aaaa.us-east4-0.gcp.cloud.qdrant.io:6333", 
#     api_key="",
# )

# from langchain_community.llms import HuggingFaceHub
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain_community.llms import HuggingFaceTextGenInference

# from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings()
# query_result = embeddings.embed_query(subtitles_text)
# print(query_result)

# inference_api_key = ""
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

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