import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Set the recording parameters
duration = 5  # seconds
sample_rate = 44100  # adjust based on your microphone specifications

# Record audio
print("Recording...")
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
print("Recording done.")

# Save the recorded audio to a WAV file
wav.write('reference_audio.wav', sample_rate, audio_data)

print("Audio saved to test.wav.")
