import librosa
import numpy as np
import pyaudio

def record_audio(seconds=15, sample_rate=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")

    frames = []
    for _ in range(int(sample_rate / 1024 * seconds)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Recording done.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames)

def compare_audio(live_audio, reference_audio):
    # Use librosa to compute cross-correlation
    cross_corr = np.correlate(live_audio, reference_audio, mode='full')

    # Find the index of the maximum correlation
    max_corr_index = np.argmax(cross_corr)

    return max_corr_index

# Replace 'reference_audio.wav' with the path to your reference audio file
reference_file = 'reference_audio.wav'

# Load the reference audio file
reference_audio, _ = librosa.load(reference_file, sr=None, mono=True)

# Record live audio (adjust the recording time as needed)
live_audio = record_audio(seconds=5)

# Compare live audio with reference audio
correlation_index = compare_audio(live_audio, reference_audio)

print("Maximum correlation index:", correlation_index)
