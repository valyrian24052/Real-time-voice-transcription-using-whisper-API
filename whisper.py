import os
import numpy as np
import speech_recognition as speech_rec
import whisper
import torch
from datetime import datetime, timedelta
from queue import Queue
from time import sleep

def main():
    # Configuration parameters
    chosen_model = "medium"
    energy_threshold_level = 1000
    record_timeout_seconds = 2
    phrase_timeout_seconds = 3

    # Initialize microphone, audio model, and recorder
    microphone_source = speech_rec.Microphone(sample_rate=16000, device_index=1)
    model_name = chosen_model + ".en" if chosen_model != "large" else chosen_model
    audio_model = whisper.load_model(model_name)
    audio_recorder = speech_rec.Recognizer()
    audio_recorder.energy_threshold = energy_threshold_level
    audio_recorder.dynamic_energy_threshold = False
    data_buffer_queue = Queue()

    def audio_record_callback(_, audio: speech_rec.AudioData) -> None:
        # Callback to handle incoming audio data
        raw_audio_data = audio.get_raw_data()
        data_buffer_queue.put(raw_audio_data)

    audio_recorder.listen_in_background(microphone_source, audio_record_callback, phrase_time_limit=record_timeout_seconds)

    print("Speech Model loaded.\n")

    # Variables for transcription handling
    transcription_lines = ['']
    last_phrase_time = None

    while True:
        try:
            current_time = datetime.utcnow()

            # Check if there's audio data in the buffer
            if not data_buffer_queue.empty():
                is_phrase_complete = False

                # Check if the time between phrases has exceeded the limit
                if last_phrase_time and current_time - last_phrase_time > timedelta(seconds=phrase_timeout_seconds):
                    is_phrase_complete = True

                last_phrase_time = current_time

                # Process the audio data
                combined_audio_data = b''.join(data_buffer_queue.queue)
                data_buffer_queue.queue.clear()
                audio_np_data = np.frombuffer(combined_audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe the audio data
                result = audio_model.transcribe(audio_np_data, fp16=torch.cuda.is_available())
                transcribed_text = result['text'].strip()

                # Update transcription based on completeness of the phrase
                if is_phrase_complete:
                    transcription_lines.append(transcribed_text)
                else:
                    transcription_lines[-1] = transcribed_text

                # Clear console and print transcriptions
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription_lines:
                    print(line)
                print('', end='', flush=True)
                sleep(0.25)

        except KeyboardInterrupt:
            break

    # Final transcription output
    print("\n\nTranscription:")
    for line in transcription_lines:
        print(line)

if __name__ == "__main__":
    main()
