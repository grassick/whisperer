# Whisperer

Push-to-talk dictation. Hold a key, speak, release, and the transcript is typed into the active window.

By default, speech is transcribed locally on the CPU with [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (English), which takes a few hundred milliseconds per clip. It can also use the OpenAI Whisper API instead.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```
uv sync
```

On Linux (X11), also install `xclip` for the clipboard:

```
sudo apt-get install xclip
```

The first run downloads the Parakeet model (about 650 MB) into the Hugging Face cache.

For translation, or to use the OpenAI backend, create a file named `openai_api_key.txt` next to `whisperer.py` and paste your OpenAI API key into it.

### Linux on Wayland

On Wayland, Whisperer reads the keyboard directly with `evdev` and pastes through a virtual keyboard, since X11 key capture and typing only work in XWayland windows. This needs `wl-clipboard` and access to the input devices:

```
sudo apt-get install wl-clipboard
sudo usermod -aG input $USER
echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger /dev/uinput
```

Then log out and back in. On Wayland the transcript is always pasted with ctrl + v (so it replaces the clipboard, and terminals need ctrl + shift + v instead).

## Usage

```
uv run whisperer.py
```

To send audio to the OpenAI Whisper API instead of transcribing locally:

```
uv run whisperer.py --backend openai
```

To quit, press ctrl + c.

- Hold right ctrl to record, and release it to transcribe and type the result.
- Tap right shift while recording to translate the transcript to Quebec French (uses the OpenAI API).
- Say "new paragraph" to insert a blank line.
- Double-tap right ctrl before recording to paste the transcript through the clipboard instead of typing it.

## Notes

- Audio is recorded at 44100 Hz (so Bluetooth headsets like AirPods work) and downsampled to 16000 Hz for transcription.
- Recordings shorter than 1 second are ignored.
- Parakeet can't be given a vocabulary prompt like the OpenAI backend can, so common misspellings of "mWater" are corrected after transcription.
