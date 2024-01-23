from youtube_transcript_api import YouTubeTranscriptApi
video_url = "https://www.youtube.com/watch?v=o3K_HbpWNpgs"
try:
    video_id = video_url.split("v=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    if transcript:
        subtitles_text = " ".join([entry["text"] for entry in transcript])
        print(subtitles_text)
    else:
        print("TS NA")
except Exception as e:
    print(f"An error occured:{str(e)}")