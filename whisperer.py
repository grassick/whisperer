import sounddevice as sd
import numpy as np
import openai
from pynput.keyboard import Listener, Controller, Key, KeyCode
from scipy.io.wavfile import write

# Path to the file containing your OpenAI API key
api_key_path = 'openai_api_key.txt'

# Key to hold down to start recording
# record_key = Key.alt_r
record_key = Key.ctrl_r

# Key to tap turn on translation
# translate_key = KeyCode.from_char('t')
translate_key = Key.shift_r

# Initialize variables
recording = False
audio_data = []
stream = None
translate = False

# Print a nice message if the API key file isn't present.
try:
    with open(api_key_path, 'r') as file:
        pass
except FileNotFoundError:
    print(f"Please create a file called {api_key_path} and paste your OpenAI API key in it.")
    exit()

# Read the API key from the file
with open(api_key_path, 'r') as file:
    openai.api_key = file.read().strip()

# Callback function to collect audio data
def callback(indata, frames, time, status):
    global audio_data
    if recording:
        # Check if indata has the expected shape (e.g., (32,))
        if indata.shape[1] == 1:  # Check if indata is mono
          audio_data.append(indata.copy())
        else:
          # Resize indata or discard it
          pass

keyboard = Controller()

def on_press(key):
    global recording, stream, audio_data, translate

    if key == record_key:
      recording = True
      translate = False
      print("Recording started...")

      audio_data = []
      
      # Initialize and start InputStream
      stream = sd.InputStream(callback=callback, channels=1, samplerate=16000)
      stream.start()

    # If recording and the translate key is pressed, set translate to True and throw away the keypress.
    if key == translate_key:
      print("Translate key pressed.")
      translate = True
      
def on_release(key):
    global recording, stream, translate

    if key == record_key:
      recording = False
      print("Recording stopped.")

      # Stop and close InputStream
      if stream is not None:
          stream.stop()
          stream.close()
          stream = None

      if audio_data == []:
        print("No audio data recorded.")
        return
      
      # Concatenate all audio data into one NumPy array
      audio_data_np = np.concatenate(audio_data, axis=0)

      # Get length of audio data in seconds
      audio_data_length = len(audio_data_np) / 16000

      if audio_data_length < 1:
        print("Audio data is less than 1 second long.")
        return

      # Write audio data to file
      write("output.wav", 16000, audio_data_np)
      
      # Save or send the audio_data_np to OpenAI Whisper
      with open("output.wav", "rb") as file:
        print("Sending audio data to OpenAI Whisper...")
        transcript = openai.Audio.transcribe("whisper-1", file, prompt="How are you doing today? I'm really looking forward to seeing you again!")
        transcript_text = transcript["text"]

        # Replace "New paragraph." with "\n"
        transcript_text = transcript_text.replace("New paragraph.", "\n\n")

        print("Transcript:")
        print(transcript_text)

        if translate:
          translate = False

          print("Translating transcript to French...")
          result = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
              {"role": "system", "content": "You translate the input text to Quebec French. You only output the text and nothing else."},
              {"role": "user", "content": transcript_text},
            ]
          )

          transcript_text = result["choices"][0]["message"]["content"]
          print(transcript_text)

        keyboard.type(transcript_text)
      

# Start listening for key events
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
