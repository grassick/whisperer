# Whisperer

Whisperer is a Python application that records audio when a specific key is held down, and sends the audio to OpenAI's Whisper ASR system for transcription when another key is tapped. It then types the transcription into the active window.

## Prerequisites

- Python 3
- OpenAI API key

## Dependencies

- sounddevice
- numpy
- openai
- pynput
- scipy

You can install these dependencies using pip:

```
pip install sounddevice numpy openai pynput scipy
```

## Setup

1. Clone the repository.
2. Create a file named openai_api_key.txt in the root directory of the project.
3. Paste your OpenAI API key into openai_api_key.txt.


## Usage

Run whisperer.py to start the application.

```
python whisperer.py
```

To quit the application, press ctrl + c.

- Hold down right ctrl button to start recording audio.
- Release right ctrl button to stop recording audio.

If you want to translate the recorded audio to French, tap the right shift button while recording

## Notes

- The audio is recorded at a sample rate of 16000 Hz and saved as output.wav.
- The application only records while the record key is held down.
- The application only sends audio to Whisper when the translate key is tapped.
- The application does not transcribe audio that is less than 1 second long.
- The application does not handle errors from the Whisper API.
