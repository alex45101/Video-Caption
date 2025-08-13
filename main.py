import os
import json
import ffmpeg
import numpy as np
from faster_whisper import WhisperModel
from moviepy import TextClip, CompositeVideoClip, ColorClip

CURR_DIR = os.getcwd()
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

    return None

def write_json_data(json_filename, data):
    """
    Write JSON data to a file

    Args:
        json_filename (str): Path to the JSON file.

    Returns:
        None
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

    audio_filename = mp4_file.replace('.mp4', '.mp3')

    if not os.path.exists(mp4_file):
        print(F"Error: '{mp4_file}' not found")
        return None

    try:
        video_stream = ffmpeg.input(mp4_file)
            
        audio = video_stream.audio

        audio_stream = ffmpeg.output(audio, audio_filename, acodec='mp3')
        audio_stream = ffmpeg.overwrite_output(audio_stream)
        audio_stream.run()
        
        print(f"MP3 file generated: {audio_filename}")
        return audio_filename
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



def main():
    """
    Main function to load configuration, process video/audio, and generate subtitles.
    """

    data = load_json_data(JSON_INFO)

    if data is None:
        return
    
    video_filename = data['Filename']   
    subtitle_data = data['Subtitle Info'] 

    audio_filename = convert_mp3_to_mp4(video_filename)
    
    set_raw_output(audio_filename)
    output_data = load_json_data(JSON_RAW_OUTPUT)

    modified_output_data = combine_words(output_data, subtitle_data['Max Chars'], subtitle_data['Max Duration'], subtitle_data['Max Gap'])
    write_json_data(JSON_MODIFIED_OUTPUT, modified_output_data)
   

if __name__ == '__main__':
    main()
