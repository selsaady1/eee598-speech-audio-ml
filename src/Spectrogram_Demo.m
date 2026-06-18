% Spectrogram and Frequency Analysis of an Audio Signal
%
% This script demonstrates how to compute and visualize the spectrogram of an audio signal,
% as well as perform frequency analysis at specific time points and frequencies.
%
% The key feature of this analysis is the **spectrogram**, which represents the frequency content of
% the audio signal over time. It provides a time-frequency representation of the signal,
% allowing us to examine how the frequency components of the signal evolve over time.
%
% The script includes:
% - **Spectrogram computation** using overlapping frames and a Hamming window to reduce edge effects.
% - **Visualization of the spectrogram**, where the color scale represents the magnitude of frequency components in dB.
% - **Frequency spectrum at a specific time point** (0.435 seconds), showing the frequency content at that moment.
% - **Amplitude evolution at a specific frequency** (540 Hz), displaying how the amplitude of this frequency changes over time.
%
% You should observe:
% - The spectrogram provides a clear representation of the dominant frequencies in the signal over time.
% - The frequency spectrum at 0.435 seconds shows which frequencies are active at that exact moment.
% - The amplitude vs. time plot at 540 Hz shows how this particular frequency behaves throughout the signal.

% Clear workspace, command window, and close all figures
clear
clc
close all

% Load the audio file 'sa1.wav' and store it in variable x
% fs is the sampling frequency of the audio signal
[x, fs] = audioread('sa1.wav');

% Set the frame length to 20 ms
fr_len = 20;  % Frame length in milliseconds

% Convert the frame length to samples
fr_N = ((fr_len/1000) * fs);  % Number of samples per frame

% Set the frame shift to 1/5th of the frame length
shift_R = fr_N / 5;  % Frame shift (overlap)

% Define a Hamming window of the same length as the frame
w = window(@hamming, fr_N);

% Compute the spectrogram of the audio signal
% S - Spectrogram matrix (complex values)
% F - Frequency vector (in Hz)
% T - Time vector (in seconds)
% P - Power spectral density (not used in this case)
[S, F, T, P] = spectrogram(x, w, fr_N - shift_R, fr_N, fs);

% Plot the spectrogram using imagesc (in dB scale)
figure
imagesc(T, F, 20*log10(abs(S)));  % Convert magnitude of S to dB scale
axis xy                         % Set the axes to normal orientation (x: time, y: frequency)
xlabel('Time (s)')              % Label for x-axis (Time)
ylabel('Frequency (Hz)')        % Label for y-axis (Frequency)
title('Spectrogram of the Audio Signal')

% Plot the frequency spectrum at a specific time (0.435 s)
figure
[temp1, temp2] = min(abs(T - 0.435));  % Find the time index closest to 0.435 seconds
plot(F, 20*log10(abs(S(:, temp2))));   % Plot the frequency content at this time
xlabel('Frequency (Hz)')               % Label for x-axis (Frequency)
ylabel('Magnitude (dB)')               % Label for y-axis (Magnitude in dB)
title('Frequency Spectrum at 0.435 seconds')

% Plot the temporal evolution of the amplitude at a specific frequency (540 Hz)
figure
[temp1, temp2] = min(abs(F - 540));    % Find the frequency index closest to 540 Hz
plot(T, abs(S(temp2, :)));             % Plot the magnitude over time at this frequency
xlabel('Time (s)')                     % Label for x-axis (Time)
ylabel('Amplitude')                    % Label for y-axis (Amplitude)
title('Amplitude vs Time at 540 Hz')