% Overlap-Add Demo for Windowed Signal Processing
%
% This script demonstrates the overlap-add method using different window functions
% to analyze and reconstruct segments of an audio signal. The overlap-add method 
% splits the signal into overlapping frames, applies a window function to each frame, 
% and then sums the frames to reconstruct the signal.
%
% Key steps in the process:
% - The signal is divided into frames of 20 ms with 25% overlap between consecutive frames.
% - A window function (e.g., triangular, Hamming, or rectangular) is applied to each frame.
% - The frames are summed back together using overlap-add.
%
% This demo visualizes:
% 1. The sum of the applied windows across frames.
% 2. The reconstructed signal after applying the window function and summing the frames.
%
% You should observe:
% - The effect of the window function on the overlap region between frames.
% - How the signal is reconstructed through the overlap-add method.

% Clear the workspace, command window, and close all figures
clear
clc
close all

% Load an audio file 'sa1.wav'
[x, fs] = audioread('sa1.wav');

% Set the frame length to 20 ms
fr_len = 20;               % Frame length in milliseconds
fr_N = ((fr_len/1000) * fs); % Convert frame length to number of samples

% Set the frame shift to 25% of the frame length
shift_R = 0.25 * fr_N;      % Frame shift (overlap is 75%)

% Initialize an array to store the summed windows
sum_w = zeros(11 * fr_N, 1);  % For plotting the sum of windows (11 frames)

% First visualization: sum of the applied windows
figure(1)
hold on

% Loop through the signal using overlapping frames
for i = 1:shift_R:10 * fr_N
    n = [i:i + fr_N - 1];    % Define the frame indices
    
    % Apply a window function (triangular in this case)
    w = window(@triang, fr_N);
    % You can switch to Hamming or rectangular window by uncommenting the lines below:
    % w = window(@hamming, fr_N);
    % w = window(@rectwin, fr_N);
    
    % Add the current window to the sum_w array
    sum_w(n) = sum_w(n) + w;

    % Plot the signal as it is reconstructed
    plot(sum_w)
    pause(0.1)
end

% Plot the sum of the applied windows (visualizing the overlapping windows)
figure(2)
plot(sum_w)
title('Sum of Applied Windows')
xlabel('Samples')
ylabel('Amplitude')

% Pause to allow visualization of the first plot
pause

% Reinitialize the summed array for the signal reconstruction
sum_w = zeros(100 * fr_N, 1);  % For reconstructing the signal (100 frames)

% Loop through the signal again with overlapping frames for signal reconstruction
for i = 1:shift_R:99 * fr_N
    n = [i:i + fr_N - 1];    % Define the frame indices
    
    % Apply the same window function to the frame
    w = window(@triang, fr_N);  % Triangular window
    % Multiply the signal segment by the window function
    xwin = x(n) .* w;
    
    % Add the windowed signal segment to the sum_w array
    sum_w(n) = sum_w(n) + xwin;
end

% Plot the reconstructed signal using overlap-add method
figure(3)
plot(sum_w)
title('Reconstructed Signal via Overlap-Add')
xlabel('Samples')
ylabel('Amplitude')