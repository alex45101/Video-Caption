import os
import json
import ffmpeg
import numpy as np
from faster_whisper import WhisperModel
from moviepy.editor import TextClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip, ColorClip

JSON_INFO = 'info.json'
JSON_RAW_OUTPUT = 'output.json'
JSON_MODIFIED_OUTPUT = 'modifiedOutput.json'

def load_json_data(json_filename):
    """
    Loads JSON data from a file.

    Args:
        json_filename (str): Path to the JSON file.

    Returns:
        dict or None: Parsed JSON data if successful, None otherwise.
    """
    try:
        with open(json_filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: '{json_filename}' not found")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_filename}'")
    # Return None if file not found or JSON is invalid
    return None

def write_json_data(json_filename, data):
    """
    Writes JSON data to a file.

    Args:
        json_filename (str): Path to the JSON file.
        data (dict): Data to write to the file.
    """
    with open(json_filename, 'w') as file:
        json.dump(data, file, indent=4)

def convert_mp3_to_mp4(mp4_file):
    """
    Extracts audio from an MP4 video file and saves it as an MP3 file.

    Args:
        mp4_file (str): Path to the MP4 video file.

    Returns:
        str or None: Path to the generated MP3 file, or None if failed.
    """
    audio_file_path = mp4_file.replace('.mp4', '.mp3')

    # Check if the input video file exists
    if not os.path.exists(mp4_file):
        print(f"Error: '{mp4_file}' not found")
        return None

    try:
        # Use ffmpeg to extract audio
        video_stream = ffmpeg.input(mp4_file)
        audio = video_stream.audio
        audio_stream = ffmpeg.output(audio, audio_file_path, acodec='mp3')
        audio_stream = ffmpeg.overwrite_output(audio_stream)
        audio_stream.run()
        print(f"MP3 file generated: {audio_file_path}")
        return audio_file_path
    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e}")
        return None
            
def set_raw_output(audio_filename, model_size = 'medium'):
    """
    Transcribes audio using WhisperModel and saves word-level info to a JSON file.

    Args:
        audio_filename (str): Path to the audio file to transcribe.

    Returns:
        None
    """
    model = WhisperModel(model_size)

    segments, info = model.transcribe(audio_filename, word_timestamps=True)

    word_info = []
    for segment in segments:
        for word in segment.words:
            word_info.append({'start': float(word.start), 'end': float(word.end), 'word': word.word})

    write_json_data(JSON_RAW_OUTPUT, word_info)    

def combine_words(data, max_chars = 30, max_duration = 2.5, max_gap = 1.5):
    """
    Combines words into subtitle lines based on character, duration, and gap constraints.

    Args:
        data (list): List of word dictionaries with 'start', 'end', and 'word' keys.
        max_chars (int): Maximum number of characters per subtitle line.
        max_duration (float): Maximum duration (in seconds) per subtitle line.
        max_gap (float): Maximum allowed gap (in seconds) between words in the same line.

    Returns:
        list: List of subtitle line dictionaries with 'start', 'end', and 'line' keys.
    """
    subtitle_lines = []

    current_line = {
        'text': "",
        'start': None,
        'end': None,
        'duration': 0.0,
        'words': []        
    }

    for i, word_data in enumerate(data):         
        word_text = word_data['word']    
        word_start = word_data['start']
        word_end = word_data['end']
        word_duration = word_end - word_start

        #calculate gap from previous word
        gap = 0
        if i > 0:
            gap = word_start - data[i - 1]['end']

        #Checking if any constraints have been hit
        max_time_hit = (current_line['duration'] + word_duration) > max_duration
        max_chars_hit = (len(current_line['text']) + len(word_text)) >= max_chars
        max_gap_hit = gap > max_gap
            
        if current_line['text'] and (max_time_hit or max_chars_hit or max_gap_hit):            
            subtitle_lines.append({
                'start': current_line['start'], 
                'end': current_line['end'], 
                'line': current_line['text'].strip(),
                'words': current_line['words']
                })
            
            current_line = {
                'text': "",
                'start': None,
                'end': None,
                'duration': 0.0,
                'words': []
            }
      
        #Start time for the new line if needed
        if current_line['start'] is None:
            current_line['start'] = word_start

        current_line['words'].append({
            'text': word_text,
            'start': word_start,
            'end': word_end
        }) 
        current_line['text'] += word_text
        current_line['end'] = word_end
        current_line['duration'] += word_duration

    #Add any remaining text as the last subtitle line
    if current_line['text']:
        subtitle_lines.append({
            'start': current_line['start'], 
            'end': current_line['end'], 
            'line': current_line['text'].strip(),
            'words': current_line['words']
        })
    
    return subtitle_lines

def create_caption_clip(caption_line_data, video_size, font = "Arial", font_size = 120, color = 'white', stroke_color = None, stroke_width = 1, caption_position = None):        
    """
    Creates a TextClip for a single caption line with specified styling and timing.

    Args:
        caption_word_data (dict): Dictionary containing 'line', 'start', and 'end' keys for the caption.
        frame_size (tuple): (width, height) of the video frame.
        font (str): Font name for the caption text.
        font_size (int): Font size for the caption text.
        color (str): Text color.
        stroke_color (str, optional): Outline color for the text.
        stroke_width (int, optional): Outline thickness for the text.
        caption_position (tuple, optional): (x, y) position for the caption. Defaults to bottom center.

    Returns:
        TextClip: The configured caption clip.
    """
    video_width, video_height = video_size[0], video_size[1]

    if caption_position is None:
        caption_position = ('center', video_height * 3/4)

    full_duration = caption_line_data['end'] - caption_line_data['start']

    caption_clip = TextClip(
        caption_line_data['line'],
        font=font,
        fontsize=font_size,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        color=color
    ).set_start(caption_line_data['start']).set_duration(full_duration)

    caption_clip = caption_clip.set_position(caption_position)

    return caption_clip

def create_caption(caption_data, frame_size, subtitle_data):
    """
    Creates a list of TextClips for all caption lines in the video.

    Args:
        caption_data (list): List of dictionaries, each containing 'line', 'start', and 'end' keys for a caption.
        frame_size (tuple): (width, height) of the video frame.
        subtitle_data (dict): Subtitle style and configuration options (font, size, color, etc.).

    Returns:
        list: List of TextClip objects for each caption line.
    """
    caption_clips = []
    for caption in caption_data:
        caption_clips.append(create_caption_clip(
            caption_line_data=caption,
            video_size=frame_size,
            font=subtitle_data['Font'],
            font_size=subtitle_data['Font Size'],
            color=subtitle_data['Color'],
            stroke_color=subtitle_data['Stroke Color'],
            stroke_width=subtitle_data['Stroke Width']
        ))
    return caption_clips


def main():
    """
    Main function to load configuration, process video/audio, and generate subtitles.
    """

    """
    dummy_clip = TextClip('Dummy Text')
    available_fonts = dummy_clip.list('font')
    for font in available_fonts:
        print(font)

    """    
    #Get data for subtitle info
    config_data = load_json_data(JSON_INFO)

    if config_data is None:
        return
    
    video_file_path = config_data['Filename']   
    subtitle_data = config_data['Subtitle Info'] 

    audio_filename = convert_mp3_to_mp4(video_file_path)

    #Output the audio file to JSON and store it
    set_raw_output(audio_filename)
    transcription_data = load_json_data(JSON_RAW_OUTPUT)

    #Combine words based on subtitle info
    processed_subtitles = combine_words(
                                    transcription_data, 
                                    subtitle_data['Max Chars'], 
                                    subtitle_data['Max Duration'], 
                                    subtitle_data['Max Gap']
                                )
    
    
    write_json_data(JSON_MODIFIED_OUTPUT, processed_subtitles)
    processed_subtitles = load_json_data(JSON_MODIFIED_OUTPUT)

    #Create caption for video
    video_clip = VideoFileClip(video_file_path)
    video_size = video_clip.size
    
    caption_clips = create_caption(processed_subtitles, video_size, subtitle_data) 

    #Output the new video
    final_video_clip = CompositeVideoClip([video_clip] + caption_clips)
    final_video_clip.write_videofile('output.mp4')    

if __name__ == '__main__':
    main()
