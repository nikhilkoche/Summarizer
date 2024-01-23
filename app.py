from langchain_community.document_loaders import YoutubeLoader
data = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=QsYGlZkevEg", add_video_info=False
)
if len(data.load()) == 0:
    print('the list is empty')