% Fourier Transform and Signal Analysis Demo
%
% This script demonstrates the use of the Fourier Transform to analyze the frequency content of various signals.
% It starts with a simple cosine wave and progresses to more complex signals, including composite signals and an audio file.
%
% Key steps in this process:
% 1. **Single cosine wave analysis**:
%    - Generate a cosine wave of 500 Hz and compute its Fourier Transform.
%    - Visualize the magnitude spectrum in both angular frequency (radians/sample) and standard frequency (Hz).
% 2. **Composite signal analysis**:
%    - Create a signal composed of multiple cosine waves with different frequencies.
%    - Compute and visualize the Fourier Transform of the composite signal to identify the frequency components.
% 3. **Time-domain and frequency-domain analysis of an audio signal**:
%    - Load an audio signal and visualize it in both the time and frequency domains using the Fourier Transform.
%
% Students should observe:
% - The clear frequency peaks in the magnitude spectrum for single and composite cosine waves, corresponding to their respective frequencies.
% - How the Fourier Transform decomposes complex signals into their individual frequency components.
% - The transition from time-domain to frequency-domain representations of real-world signals like audio, offering insights into their frequency content.
%
% This demonstration provides a comprehensive introduction to Fourier analysis, highlighting its application for understanding the frequency structure of signals.



% Clear the workspace and command window
clear
clc

% Set the signal frequency (in Hz) and sampling frequency (in Hz)
f = 500;          % Frequency of the signal (500 Hz)
fs = 8000;        % Sampling frequency (8000 Hz)

% Calculate the angular frequency (radians/sample)
w = 2*pi*f/fs;

% Define the length of the signal (in seconds)
len = 2;          % Signal duration (2 seconds)

% Create a time index vector (samples) for the entire signal length
n = [1:fs*len];   % Index of time samples

% Convert the time index to actual time values (in seconds)
t = n/fs;         % Time vector

% Generate a cosine wave with frequency f
x = cos((2*pi*f/fs)*n);

% Compute the magnitude of the Fourier Transform (in dB) of the signal
XX = 20*log10(abs(fft(x)));

% Plot the magnitude spectrum of the signal (in dB)
figure
plot(XX)
title('Magnitude Spectrum of Cosine Wave')
xlabel('Frequency Index')
ylabel('Magnitude (dB)')

% Calculate the frequency step (delta omega) for plotting purposes
delw = (2*pi)/(fs*len);

% Create the angular frequency vector for plotting (radians/sample)
w = n*delw;

% Plot the magnitude spectrum against the angular frequency (radians/sample)
figure
plot(w, XX)
title('Magnitude Spectrum vs Angular Frequency')
xlabel('Angular Frequency (rad/sample)')
ylabel('Magnitude (dB)')

% Convert the angular frequency to frequency in Hz
f = (w*fs)/(2*pi);

% Plot the magnitude spectrum in Hz, only for the first half (positive frequencies)
figure
plot(f(1:len/2*fs), XX(1:len/2*fs))
title('Magnitude Spectrum (Hz)')
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')

% Generate a new composite signal with multiple cosine waves of different frequencies
x = .8*cos((2*pi*200/fs)*n) + 4*cos((2*pi*250/fs)*n) + ...
    3*cos((2*pi*350/fs)*n) + 4*cos((2*pi*600/fs)*n);

% Compute the Fourier Transform of the composite signal (no log scaling)
XX = abs(fft(x));

% Plot the magnitude spectrum of the composite signal (in Hz, only positive frequencies)
figure
plot(f(1:len/2*fs), XX(1:len/2*fs))
title('Magnitude Spectrum of Composite Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')

% Plot the time-domain signal (composite of multiple cosines)
figure
plot(t, x)
title('Time-Domain Signal (Composite)')
xlabel('Time (seconds)')
ylabel('Amplitude')

% Load an audio file 'sa1.wav' (assuming it's in the working directory)
[x, fs] = audioread('sa1.wav');

% Create the time vector for the audio signal
t = [1:length(x)]/fs;

% Create the frequency vector for the audio signal
f = linspace(0, fs, length(x));

% Plot the audio signal in the time domain
figure
plot(t, x)
title('Time-Domain Audio Signal')
xlabel('Time (seconds)')
ylabel('Amplitude')

% Compute the Fourier Transform of the audio signal (magnitude only)
XX = abs(fft(x));

% Plot the magnitude spectrum of the audio signal (only for positive frequencies)
figure
plot(f(1:length(x)/2), XX(1:length(x)/2))
title('Magnitude Spectrum of Audio Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')