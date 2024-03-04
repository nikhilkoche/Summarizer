from youtube_transcript_api import YouTubeTranscriptApi
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer
from semantic_text_splitter import TiktokenTextSplitter
from transformers import BartForConditionalGeneration, BartTokenizer

#load the model and tokenizer

model = BartForConditionalGeneration.from_pretrained(
    'facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

#get transcript for YouTube using YouTubeApi
def get_transcript(url):
    video_id = url.split("v=")[1].split("&")[0]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript



#summarization model
def summarize(text, maxSummarylength=500):
    # Encode the text and summarize
    inputs = tokenizer.encode("summarize: " +
                              text,
                              return_tensors="pt",
                              max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=maxSummarylength,
                                 min_length=int(maxSummarylength/5),
                                 length_penalty=10.0,
                                 num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



def split_text_into_pieces(text,
                           max_tokens=900,
                           overlapPercent=10):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Calculate the overlap in tokens
    overlap_tokens = int(max_tokens * overlapPercent / 100)

    # Split the tokens into chunks of size
    # max_tokens with overlap
    pieces = [tokens[i:i + max_tokens]
              for i in range(0, len(tokens),
                             max_tokens - overlap_tokens)]

    # Convert the token pieces back into text
    text_pieces = [tokenizer.decode(
        tokenizer.convert_tokens_to_ids(piece),
        skip_special_tokens=True) for piece in pieces]

    return text_pieces



def recursive_summarize(text, max_length=200, recursionLevel=0):
    recursionLevel=recursionLevel+1
    print("######### Recursion level: ",
          recursionLevel,"\n\n######### ")
    tokens = tokenizer.tokenize(text)
    expectedCountOfChunks = len(tokens)/max_length
    max_length=int(len(tokens)/expectedCountOfChunks)+2

    # Break the text into pieces of max_length
    pieces = split_text_into_pieces(text, max_tokens=max_length)

    print("Number of pieces: ", len(pieces))
    # Summarize each piece
    summaries=[]
    k=0
    for k in range(0, len(pieces)):
        piece=pieces[k]
        print("****************************************************")
        print("Piece:",(k+1)," out of ", len(pieces), "pieces")
        print(piece, "\n")
        summary =summarize(piece, maxSummarylength=max_length/3*2)
        print("SUMNMARY: ", summary)
        summaries.append(summary)
        print("****************************************************")

    concatenated_summary = ' '.join(summaries)

    tokens = tokenizer.tokenize(concatenated_summary)

    if len(tokens) > max_length:
        # If the concatenated_summary is too long, repeat the process
        print("############# GOING RECURSIVE ##############")
        return recursive_summarize(concatenated_summary,
                                   max_length=max_length,
                                   recursionLevel=recursionLevel)
    else:
      # Concatenate the summaries and summarize again
        final_summary=concatenated_summary
        if len(pieces)>1:
            final_summary = summarize(concatenated_summary,
                                  maxSummarylength=max_length)
        return final_summary



##remove later
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