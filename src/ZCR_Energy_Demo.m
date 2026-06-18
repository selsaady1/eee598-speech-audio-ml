% Energy and Zero-Crossing Rate (ZCR) Analysis of an Audio Signal
%
% This script demonstrates how to compute and visualize two key features of an audio signal:
% 1. **Energy**: The total signal power within each frame, which reflects the signal's amplitude.
% 2. **Zero-Crossing Rate (ZCR)**: The rate at which the signal crosses the zero line, useful for distinguishing voiced and unvoiced speech.
%
% The script processes the audio signal in small overlapping frames:
% - Each frame is 20 ms long, with a 50% overlap between consecutive frames.
% - The signal is windowed using a Hamming window to reduce edge effects.
% 
% For each frame:
% - **Energy** is computed by squaring the signal values and summing the result.
% - **ZCR** is computed by counting how many times the signal changes sign (crosses zero).
%
% The script plots three graphs:
% - The normalized **original audio signal**.
% - The normalized **energy** across time, highlighting periods of higher amplitude (such as voiced segments).
% - The normalized **ZCR**, which typically shows higher values for unvoiced speech or noise.
%
% You should observe the following:
% - **Energy** tends to be higher in voiced segments (where there is significant speech activity).
% - **ZCR** is higher in unvoiced segments or noise due to more frequent zero-crossings.
% 
% This analysis provides insights into the temporal characteristics of speech and can be used in speech processing tasks such as voice activity detection or speech classification.



% Clear the workspace and command window
clear
clc

% Load the audio file 'sa1.wav' and store it in variable x
% fs is the sampling frequency of the audio signal
[x, fs] = audioread('sa1.wav');

% Set the frame length to 20 ms
fr_len = 20;  % Frame length in milliseconds

% Play the loaded audio signal
sound(x, fs);

% Convert frame length to samples
fr_N = ((fr_len / 1000) * fs);  % Number of samples per frame

% Set the frame shift to half the frame length (50% overlap)
shift_R = fr_N / 2;  % Frame shift (overlap in samples)

% Initialize arrays to store energy and zero-crossing rate (ZCR)
E = [];    % Energy for each frame
zcr = [];  % Zero Crossing Rate for each frame

% Loop through the signal in frames with overlap
for i = 1:shift_R:(length(x) - fr_N)
    % Define the indices for the current frame
    n = [i:i + fr_N - 1];
    
    % Apply a Hamming window to the current frame
    w = window(@hamming, fr_N);
    
    % Calculate the energy of the current windowed frame
    % Square the signal, apply the window, and sum the result
    xwin = (x(n).^2) .* w;  % Windowed and squared signal
    E(end+1) = sum(xwin);   % Energy is the sum of the windowed squared signal
    
    % Calculate the zero-crossing rate for the current frame
    % Count the number of times the signal crosses the zero line
    xwin2 = x(n);  % Non-windowed signal for ZCR calculation
    zcr(end+1) = sum(xwin2(1:end-1) .* xwin2(2:end) < 0);  % Zero-crossings count
end

% Plot the original normalized audio signal
figure
plot(x / max(x))
hold on

% Plot the normalized energy over time
plot([1:shift_R:(length(x)-fr_N)], E / max(E), 'r')

% Plot the normalized zero-crossing rate (ZCR) over time
plot([1:shift_R:(length(x)-fr_N)], zcr / max(zcr), 'k')

% Add legend to indicate the plotted signals
legend('Speech', 'Energy', 'ZCR');