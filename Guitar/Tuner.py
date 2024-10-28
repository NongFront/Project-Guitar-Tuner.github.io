import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

# General settings that can be changed by the user
SAMPLE_FREQ = 48000  # sample frequency in Hz
WINDOW_SIZE = 48000  # window size of the DFT in samples
WINDOW_STEP = 12000  # step size of window
NUM_HPS = 5  # max number of harmonic product spectrums
POWER_THRESH = 1e-6  # tuning is activated if the signal power exceeds this threshold
WHITE_NOISE_THRESH = 0.2  # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ  # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ  # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE  # frequency step width of the interpolated DFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

# Define the target frequencies for each guitar string
GUITAR_STRING_FREQS = {
    "E2": 82.41,  # Low E
    "A2": 110.00,  # A
    "D3": 146.83,  # D
    "G3": 196.00,  # G
    "B3": 246.94,  # B
    "E4": 329.63  # High E
}

HANN_WINDOW = np.hanning(WINDOW_SIZE)

def find_closest_string(pitch):
    closest_string = min(GUITAR_STRING_FREQS, key=lambda k: abs(GUITAR_STRING_FREQS[k] - pitch))
    closest_pitch = GUITAR_STRING_FREQS[closest_string]
    return closest_string, closest_pitch

def callback(indata, frames, time, status):
    """
    Callback function of the InputStream method.
    """
    # define static variables
    if not hasattr(callback, "window_samples"):
        callback.window_samples = [0 for _ in range(WINDOW_SIZE)]

    if status:
        print(status)
        return

    if any(indata):
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0]))  # append new samples
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]  # remove old samples

        # skip if signal power is too low
        signal_power = (np.linalg.norm(callback.window_samples, ord=2) ** 2) / len(callback.window_samples)
        if signal_power < POWER_THRESH:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Closest string: ...")
            return

        # avoid spectral leakage by multiplying the signal with a hann window
        hann_samples = callback.window_samples * HANN_WINDOW
        magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])

        # suppress mains hum, set everything below 62Hz to zero
        for i in range(int(62 / DELTA_FREQ)):
            magnitude_spec[i] = 0

        # calculate average energy per frequency for the octave bands and suppress everything below it
        for j in range(len(OCTAVE_BANDS) - 1):
            ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
            ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
            ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
            avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2) ** 2) / (ind_end - ind_start)
            avg_energy_per_freq = avg_energy_per_freq ** 0.5
            for i in range(ind_start, ind_end):
                magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH * avg_energy_per_freq else 0

        # interpolate spectrum
        mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS), np.arange(0, len(magnitude_spec)),
                                  magnitude_spec)
        mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)  # normalize it

        hps_spec = copy.deepcopy(mag_spec_ipol)

        # calculate the HPS
        for i in range(NUM_HPS):
            tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))], mag_spec_ipol[::(i + 1)])
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        max_ind = np.argmax(hps_spec)
        max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

        closest_string, closest_pitch = find_closest_string(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)

        # Calculate the difference and provide advice
        diff_hz = max_freq - closest_pitch
        if abs(diff_hz) <= 1:
            advice = "In tune"
        elif diff_hz < -1:
            advice = f"Tune up by {-diff_hz:.1f} Hz"
        else:
            advice = f"Tune down by {diff_hz:.1f} Hz"

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"String: {closest_string} | Detected: {max_freq} Hz | Target: {closest_pitch} Hz | {advice}")

    else:
        print('no input')

try:
    print("Starting HPS guitar tuner...")
    with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
        while True:
            time.sleep(1)  # Slower update interval
except Exception as exc:
    print(str(exc))
