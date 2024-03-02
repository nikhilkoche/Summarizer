from youtube_transcript_api import YouTubeTranscriptApi
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer
from semantic_text_splitter import TiktokenTextSplitter



def get_transcript(url):
    video_id = url.split("v=")[1].split("&")[0]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript

def text_splitter(full_transcript):
    max_tokens =400
    tokenizer = Tokenizer.from_pretrained('bert-base-uncased')
    splitter = HuggingFaceTextSplitter(tokenizer,trim_chunks=True)
    chunks = splitter.chunks(full_transcript,max_tokens)
    return chunks

    
def main():
    url = input("Enter URL:")
    full_transcript = ''
    trans=get_transcript(url)
    for entry in trans:
        full_transcript += entry['text']
    for i, chunk in enumerate(text_splitter(full_transcript)):
        print(f"CHUNK {i*1}:",chunk)
    



if __name__ == '__main__':
    main()