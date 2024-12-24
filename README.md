# Cover Song Detection System

A Python-based system that identifies potential cover songs by comparing lyrics similarity between a YouTube video and a large database of songs using natural language processing and semantic search.

## Features

- Audio extraction from YouTube videos using yt-dlp
- Speech-to-text transcription using Insanely-Fast-Whisper
- Semantic lyrics comparison using Sentence Transformers
- High-performance similarity search with FAISS
- Support for multiple languages

## Requirements

- Python 3.10+
- CUDA-compatible GPU (optional, for faster processing)
- FFmpeg
- Required Python packages:
  - yt-dlp
  - transformers
  - torch
  - faiss-cpu (or faiss-gpu)
  - sentence-transformers
  - numpy
  - pandas
  - kagglehub

## Installation

1. Install system dependencies:
```bash
apt-get update && apt-get install -y python3.10-venv ffmpeg
```

2. Install Python packages:
```bash
pip install --upgrade pip
pip install pipx
pipx ensurepath
pip install yt_dlp transformers torch faiss-cpu sentence-transformers numpy pandas kagglehub
pipx install git+https://github.com/Vaibhavs10/insanely-fast-whisper.git
```

## Usage

1. Initialize the system:
```python
from cover_detection import load_and_preprocess_dataset, create_and_save_index

# Load and preprocess the dataset
dataset_directory = kagglehub.dataset_download("carlosgdcj/genius-song-lyrics-with-language-information")
csv_file_path = os.path.join(dataset_directory, 'song_lyrics.csv')
top_views_df = load_and_preprocess_dataset(csv_file_path, top_n=1000)

# Create and save the index
create_and_save_index(top_views_df)
```

2. Find potential covers:
```python
from cover_detection import get_covers

youtube_url = 'https://www.youtube.com/watch?v=example'
matches = get_covers(youtube_url, k=5)

# Print results
for match in matches:
    print(f"{match['Title']} by {match['Artist']} (Score: {match['Score']})")
```

## How It Works

1. **Audio Extraction**: Downloads audio from YouTube videos using yt-dlp
2. **Transcription**: Converts audio to text using Insanely-Fast-Whisper
3. **Semantic Encoding**: 
   - Cleans and preprocesses lyrics
   - Generates embeddings using the all-mpnet-base-v2 model
4. **Similarity Search**:
   - Uses FAISS for efficient similarity comparison
   - Returns top-k matches with similarity scores

## Performance

The system uses:
- GPU acceleration when available
- Efficient batch processing for embeddings
- FAISS IndexFlatIP for fast similarity search
- Chunked processing for large datasets

## Limitations

- Depends on audio quality for accurate transcription
- May struggle with heavily modified covers
- Requires significant computational resources for large-scale processing
- Limited by the accuracy of the speech-to-text model

## Contributing

Feel free to submit issues and enhancement requests through GitHub.
