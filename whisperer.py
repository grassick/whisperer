import argparse
import io
import os
import re
import selectors
import time
import sounddevice as sd
import numpy as np
import openai
import pyperclip
from pynput.keyboard import Listener, Controller, Key, KeyCode
import soundfile
from scipy import signal

# Directory containing this script, so it can be run from anywhere
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the file containing your OpenAI API key
openai_api_key_path = os.path.join(script_dir, "openai_api_key.txt")

# Key to hold down to start recording
# record_key = Key.alt_r
record_key = Key.ctrl_r

# Key to tap turn on translation
# translate_key = KeyCode.from_char('t')
translate_key = Key.shift_r

# Prompt to help the OpenAI backend with style and vocabulary (Parakeet doesn't support prompts)
transcription_prompt = "How are you doing today? I'm really looking forward to seeing you again! I work for mWater..."

# Record at 44100 Hz (AirPods compatible) and downsample to 16000 Hz for transcription
record_sample_rate = 44100
sample_rate = 16000

parser = argparse.ArgumentParser(description="Hold right ctrl to dictate into the active window.")
parser.add_argument(
    "--backend",
    choices=["parakeet", "openai"],
    default="parakeet",
    help="parakeet runs locally on the CPU; openai sends audio to the OpenAI Whisper API (default: parakeet)",
)
args = parser.parse_args()

# Initialize variables
recording = False
audio_data = []
stream = None
translate = False
# If true, the transcript will be copied to the clipboard even it doesn't contain any special characters.
force_clipboard = False

# On Wayland, pynput (X11) only sees keys when an XWayland window has focus and
# can't type into native Wayland apps, so read/write input devices via evdev instead.
use_evdev = os.environ.get("XDG_SESSION_TYPE") == "wayland"


def read_api_key(path, name):
    # Print a nice message if the API key file isn't present.
    try:
        with open(path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Please create a file called {path} and paste your {name} API key in it.")
        return None


# OpenAI client, used for translation and the openai backend. Created on first use
# so the local backend works without an OpenAI key.
openai_client = None


def get_openai_client():
    global openai_client
    if openai_client is None:
        api_key = read_api_key(openai_api_key_path, "OpenAI")
        if api_key is not None:
            openai_client = openai.OpenAI(api_key=api_key)
    return openai_client


if args.backend == "parakeet":
    import onnx_asr

    print("Loading Parakeet model...")
    # The int8 model is faster on CPU, and is a single file (the fp32 one fails to
    # load from the Hugging Face cache with newer onnxruntime)
    parakeet_model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8")
elif get_openai_client() is None:
    exit()


def transcribe(audio):
    if args.backend == "parakeet":
        return parakeet_model.recognize(audio, sample_rate=sample_rate)

    # Encode as FLAC in memory to keep the upload small
    flac = io.BytesIO()
    soundfile.write(flac, audio, sample_rate, format="flac")

    print("Sending audio data to OpenAI Whisper...")
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1", file=("audio.flac", flac.getvalue()), prompt=transcription_prompt
    )
    return transcript.text


def clean_transcript(text):
    # Replace "New paragraph." with a blank line
    text = re.sub(r"\s*\bnew paragraph\b[.,]?\s*", "\n\n", text, flags=re.IGNORECASE)

    # Parakeet can't be prompted with the spelling of mWater
    text = re.sub(r"\bM[- ]?water\b", "mWater", text, flags=re.IGNORECASE)

    return text.strip()


def translate_to_french(text):
    client = get_openai_client()
    if client is None:
        return text

    print("Translating transcript to French...")
    result = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You translate the input text to Quebec French using 'vous'. You only output the text and nothing else.",
            },
            {"role": "user", "content": text},
        ],
    )
    return result.choices[0].message.content


# Callback function to collect audio data
def callback(indata, frames, time, status):
    global audio_data
    if recording:
        if status:
            print(f"Audio callback status: {status}")
        # Check if indata has the expected shape (e.g., (frames, 1))
        if indata.shape[1] == 1:  # Check if indata is mono
            audio_data.append(indata.copy())
        else:
            # Resize indata or discard it
            pass


if use_evdev:
    from evdev import InputDevice, UInput, UInputError, ecodes, list_devices

    # Map evdev key codes to the pynput keys used by on_press/on_release
    evdev_keymap = {
        ecodes.KEY_RIGHTCTRL: Key.ctrl_r,
        ecodes.KEY_RIGHTALT: Key.alt_r,
        ecodes.KEY_RIGHTSHIFT: Key.shift_r,
    }

    # Virtual keyboard used to send CTRL-V
    try:
        virtual_keyboard = UInput(
            {ecodes.EV_KEY: [ecodes.KEY_LEFTCTRL, ecodes.KEY_V]}, name="whisperer"
        )
    except UInputError:
        print("Cannot open /dev/uinput. See README for the udev rule to allow it.")
        exit()
else:
    keyboard = Controller()


def paste_text(text):
    # Copy the transcript text to the clipboard (pyperclip uses wl-copy on Wayland)
    pyperclip.copy(text)

    # Simulate CTRL-V to paste the text
    if use_evdev:
        time.sleep(0.05)  # Give wl-copy a moment to take ownership of the clipboard
        for code, value in [
            (ecodes.KEY_LEFTCTRL, 1),
            (ecodes.KEY_V, 1),
            (ecodes.KEY_V, 0),
            (ecodes.KEY_LEFTCTRL, 0),
        ]:
            virtual_keyboard.write(ecodes.EV_KEY, code, value)
            virtual_keyboard.syn()
    else:
        keyboard.press(Key.ctrl)
        keyboard.press("v")
        keyboard.release("v")
        keyboard.release(Key.ctrl)


def output_text(text):
    global force_clipboard

    # Determine if any special characters are being used that can't be
    # typed using keyboard.type(). These are any characters that aren't in English
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-'_"
    )
    special_chars = set(text) - allowed_chars
    if use_evdev:
        # No way to type arbitrary text on Wayland, so always paste
        paste_text(text)
    elif len(special_chars) > 0 or force_clipboard:
        print("Special characters detected: " + str(special_chars))
        paste_text(text)
    else:
        # Since there are no accents, we can just use the standard type command.
        keyboard.type(text)
    force_clipboard = False


def on_press(key):
    global recording, stream, audio_data, translate

    # If recording and the translate key is pressed, set translate to True and throw away the keypress.
    if recording and key == translate_key:
        print("Translate key pressed.")
        translate = True
        return

    # Ignore all other keypresses if we're recording
    if recording:
        return

    if key != record_key:
        return

    translate = False
    recording = True
    print("Recording started...")

    audio_data = []

    stream = sd.InputStream(callback=callback, channels=1, samplerate=record_sample_rate)
    stream.start()


def on_release(key):
    global recording, stream, translate, force_clipboard

    if not recording or key != record_key:
        return

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

    # Concatenate all audio data into one mono NumPy array, downsampled to 16000 Hz
    audio = np.concatenate(audio_data, axis=0)[:, 0]
    audio = signal.resample_poly(audio, sample_rate, record_sample_rate).astype(np.float32)

    # Get length of audio data in seconds
    audio_length = len(audio) / sample_rate

    # If was start of a double-click, paste the next transcript instead of typing it
    if audio_length < 0.5:
        force_clipboard = True

    if audio_length < 1:
        print("Audio data is less than 1 second long.")
        return

    # Keep listening for keys if transcription fails (e.g. no network)
    try:
        start = time.time()
        transcript_text = clean_transcript(transcribe(audio))
        print(f"Transcript ({time.time() - start:.2f}s):")
        print(transcript_text)

        if translate:
            translate = False
            transcript_text = translate_to_french(transcript_text)
            print(transcript_text)
    except Exception as e:
        print(f"Transcription failed: {e}")
        return

    if not transcript_text:
        print("Transcript is empty.")
        return

    output_text(transcript_text)


def listen_evdev():
    # Read key events from every device that looks like a keyboard
    selector = selectors.DefaultSelector()
    for path in list_devices():
        device = InputDevice(path)
        keys = device.capabilities().get(ecodes.EV_KEY, [])
        if ecodes.KEY_A in keys:
            selector.register(device, selectors.EVENT_READ)

    if not selector.get_map():
        print("No keyboards found. Is your user in the 'input' group?")
        exit()

    while True:
        for selector_key, _ in selector.select():
            try:
                events = list(selector_key.fileobj.read())
            except OSError:
                # Device was unplugged
                selector.unregister(selector_key.fileobj)
                continue
            for event in events:
                if event.type != ecodes.EV_KEY or event.code not in evdev_keymap:
                    continue
                key = evdev_keymap[event.code]
                if event.value == 1:
                    on_press(key)
                elif event.value == 0:
                    on_release(key)
                # value 2 is autorepeat, which we ignore


print(f"Ready ({args.backend}). Hold right ctrl to dictate.")

# Start listening for key events
if use_evdev:
    listen_evdev()
else:
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
