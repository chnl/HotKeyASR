import threading
import queue
import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import keyboard
import datetime
import logging  # Added for logging
from concurrent.futures import ThreadPoolExecutor

# from faster_whisper import WhisperModel
import faster_whisper


def measure_ambient_noise(duration=1.0, samplerate=16000):
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float64"
    )
    sd.wait()
    rms = np.sqrt(np.mean(audio**2))
    return 20 * np.log10(rms) if rms > 0 else -np.inf


def adjust_vad_silence_duration(
    ambient_noise_db, base_silence_duration_ms=200, sensitivity=5
):
    adjusted_silence_duration_ms = base_silence_duration_ms + (
        ambient_noise_db * sensitivity
    )
    adjusted_silence_duration_ms = max(
        adjusted_silence_duration_ms, base_silence_duration_ms
    )
    return adjusted_silence_duration_ms


ambient_noise_db = measure_ambient_noise()
min_silence_duration_ms = adjust_vad_silence_duration(ambient_noise_db)
# Version: 1.7

# whisper_model = WhisperModel.from_pretrained("models/faster-whisper-medium")
# Initialize logging
logging.basicConfig(filename="transcription.log", level=logging.DEBUG)


def transcribe_audio(audio_data, whisper_model, transcription_count):
    start_time = datetime.datetime.now()
    logging.debug(f"Transcription {transcription_count} started at {start_time}")
    temp_audio_file = "temp_audio.wav"
    with open(temp_audio_file, "wb") as f:
        f.write(audio_data.get_wav_data())

    segments, info = whisper_model.transcribe(
        temp_audio_file,
        beam_size=2,  # Adjusting the beam size to 2 for potentially faster results
        vad_filter=True,  # Enabling VAD filter to remove long silences
        vad_parameters=dict(
            min_silence_duration_ms=min_silence_duration_ms
        )  # Using the adjusted VAD parameters
        # Customizing the VAD parameters
    )

    transcribed_text = " ".join([segment.text for segment in segments])
    end_time = datetime.datetime.now()
    latency = end_time - start_time
    logging.debug(
        f"Transcription {transcription_count} ended at {end_time} with latency {latency.total_seconds()} seconds"
    )
    return transcribed_text, latency.total_seconds()


vad_parameters_queue = queue.Queue()

recognizer = sr.Recognizer()
# Load the Faster-Whisper model from a local directory

# whisper_model = faster_whisper.WhisperModel("models/faster-whisper-medium")
whisper_model = faster_whisper.WhisperModel(
    "medium.en",
    device="cuda",
    compute_type="int8",
)


def main():
    transcription_count = 0  # Added for tracking
    try:
        with ThreadPoolExecutor() as executor:
            print("Transcription Active. Speak Now...")
            while keyboard.is_pressed("ctrl+shift"):
                with sr.Microphone(chunk_size=1024) as source:  # Adjusted buffer size
                    print("Say something:")
                    audio_data = recognizer.listen(source)
                transcription_count += 1  # Increment the count
                future = executor.submit(
                    transcribe_audio, audio_data, whisper_model, transcription_count
                )
                transcribed_text, latency = future.result()

                # Logging latency and transcription count
                logging.debug(
                    f"Transcription {transcription_count}: Latency = {latency} seconds"
                )

                print(f"Latency: {latency} seconds")
                if "quit" in transcribed_text or "exit" in transcribed_text:
                    print("Exiting...")
                    break

                print(f"Transcribed Text: {transcribed_text}")

                # Before writing the characters
                write_start_time = datetime.datetime.now()
                logging.debug(f"Writing started at {write_start_time.isoformat()}")

                # After writing the characters
                write_end_time = datetime.datetime.now()
                logging.debug(f"Writing ended at {write_end_time.isoformat()}")

                # Calculate and log the write latency
                write_latency = (write_end_time - write_start_time).total_seconds()
                logging.debug(f"Writing latency: {write_latency} seconds")
                logging.shutdown()

                # keyboard.send("ctrl+a")
                keyboard.write("# " + transcribed_text + " ")
                keyboard.send("enter")
                keyboard.write("###")

            print("Transcription Paused.")

    except KeyboardInterrupt:
        print("Exiting gracefully...")


# Function to measure ambient noise and adjust VAD parameters accordingly
def measure_and_adjust_ambient_noise():
    ambient_noise_db = measure_ambient_noise()
    logging.debug(f"Ambient noise level (dB): {ambient_noise_db}")

    # Adjust the VAD silence threshold based on ambient noise
    if ambient_noise_db < -60:  # Example threshold, needs tuning
        min_silence_duration_ms = 300
    elif -60 <= ambient_noise_db < -55:
        min_silence_duration_ms = 500
    elif -55 <= ambient_noise_db < -50:
        min_silence_duration_ms = 600
    else:
        min_silence_duration_ms = 800

    # Log the adjusted VAD parameter
    logging.debug(f"Adjusted VAD silence duration (ms): {min_silence_duration_ms}")
    return min_silence_duration_ms


# Function to periodically measure ambient noise and adjust VAD parameters
def periodic_ambient_check(interval=30):
    while True:
        min_silence_duration_ms = measure_and_adjust_ambient_noise()
        vad_parameters_queue.put(min_silence_duration_ms)
        time.sleep(interval)


# Start the periodic check in a separate thread
threading.Thread(target=periodic_ambient_check, args=(30,), daemon=True).start()


if __name__ == "__main__":
    print("Press and hold 'Ctrl+Shift' to start transcription.")
    while True:
        if keyboard.is_pressed("ctrl+shift"):
            main()
