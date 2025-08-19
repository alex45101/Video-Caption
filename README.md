# Video-Caption

A Python project for generating video subtitles using speech-to-text and overlaying captions on videos. Built with MoviePy, Faster-Whisper, and ffmpeg-python.

## Features
- Extracts audio from MP4 videos
- Transcribes speech to text using Faster-Whisper
- Automatically generates and formats subtitles
- Overlays styled captions onto the original video
- Fully configurable subtitle appearance (font, color, stroke, etc.)

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies
- [ImageMagick](https://imagemagick.org/) (required by MoviePy for text rendering)

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/alex45101/Video-Caption.git
   cd Video-Caption
   ```
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Install ImageMagick and add it to your PATH (Windows users: ensure legacy utilities are enabled).

## Usage
1. Place your video file (e.g., `IMG_0002.mp4`) in the project directory.
2. Edit `info.json` to configure subtitle options and specify the video filename.
3. Run the main script:
   ```sh
   python main.py
   ```
4. The output video with captions will be saved as `output.mp4`.

## Configuration
- `info.json`: Set subtitle appearance, timing, and video filename.
- `requirements.txt`: Minimal dependencies for the project.

## Example
```
{
    "Filename" : "IMG_0002.mp4",
    "Subtitle Info" : {
        "Max Chars" : 20,
        "Max Duration" : 2.0,
        "Max Gap" : 1.5,
        "Font" : "Segoe UI Semibold",
        "Font Size" : 120,
        "Color" : "#fdff7a",
        "Stroke Color" : "black",
        "Stroke Width" : 2,
         "Shadow" : true,
        "Shadow Color" : "black"
    }
}
```

## License
MIT License

## Credits
- [MoviePy](https://zulko.github.io/moviepy/)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)

---

Feel free to open issues or contribute!
