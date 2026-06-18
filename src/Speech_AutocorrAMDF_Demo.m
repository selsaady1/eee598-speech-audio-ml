% Autocorrelation and Average Magnitude Difference Function (AMDF) Demo
%
% This script demonstrates two methods of signal analysis: **Autocorrelation** and 
% **Average Magnitude Difference Function (AMDF)** on a voiced segment of speech.
%
% **Autocorrelation** measures the similarity of a signal with a time-shifted version of itself.
% It is commonly used to estimate the fundamental frequency (pitch) of speech.
%
% **AMDF** calculates the average magnitude of differences between the original and time-shifted signals,
% which can also be used for pitch detection but is less sensitive to noise.
%
% The script processes a voiced speech segment in three stages:
% 1. **Autocorrelation and AMDF** on the original voiced speech segment.
% 2. **Clipping** the voiced speech segment to keep only the top 70% of the amplitude and removing smaller values.
% 3. **Autocorrelation and AMDF** on the clipped speech segment.
%
% You should observe:
% - The **autocorrelation** plot exhibits periodic peaks corresponding to the periodicity of the voiced speech signal.
% - The **AMDF** shows dips where the signal and its shifted version align.
% - Clipping the signal reduces the overall energy but maintains the fundamental periodicity, which is reflected in both autocorrelation and AMDF.
%
% This demonstration helps to understand the temporal structure of voiced speech and how pitch can be estimated using these techniques.

% Clear workspace and command window
clear
clc

% Load the audio file 'sa1.wav' and extract a voiced speech segment
[x, fs] = audioread('sa1.wav');

% Extract a specific voiced speech segment
x_v = x(33920:34650);       % Select a voiced speech segment
plot(x_v)                   % Plot the voiced speech segment

% --- Autocorrelation on voiced segment ---
aco = [];                    % Initialize autocorrelation array
for i = 1:500
    % Split the signal into two parts: original and shifted versions
    sig1 = x_v(1:end-((i-1)*1));      % First part (original signal)
    sig2 = x_v(((i-1)*1)+1:end);      % Second part (shifted signal)
    
    % Plot the two signals for comparison
    subplot(211)
    plot(sig1)
    hold on
    plot(sig2, 'r')                   % Plot the shifted version in red
    hold off
    
    % Compute autocorrelation by summing pointwise multiplication of signals
    aco(i) = sum(sig1 .* sig2);
    
    % Plot the autocorrelation result
    subplot(212)
    plot(aco)
    title('Autocorrelation of Voiced Segment')
    
    % Pause to allow visualization of each iteration
    pause(0.01)
end

% Alternative: Built-in Matlab function for autocorrelation
% Uncomment the following line to use the built-in autocorrelation function
% plot(xcorr(x_v(1:500)))

% --- AMDF on voiced segment ---
amdf = [];                    % Initialize AMDF array
for i = 1:500
    % Split the signal into two parts: original and shifted versions
    sig1 = x_v(1:end-((i-1)*1));
    sig2 = x_v(((i-1)*1)+1:end);
    
    % Plot the two signals for comparison
    subplot(211)
    plot(sig1)
    hold on
    plot(sig2, 'r')                   % Plot the shifted version in red
    hold off
    
    % Compute AMDF by calculating the mean absolute difference between signals
    amdf(i) = mean(abs(sig1 - sig2));
    
    % Plot the AMDF result
    subplot(212)
    plot(amdf)
    title('AMDF of Voiced Segment')
    
    % Pause to allow visualization of each iteration
    pause(0.01)
end

% --- Clipping the voiced segment ---
% Clip the voiced speech segment to retain only the top 70% of the signal values
x_vclip = x_v;
x_vclip(and(x_v < 0.5 * max(x_v), x_v > 0)) = 0;  % Clip positive values below 50% of max
x_vclip(and(x_v > -0.5 * max(-x_v), x_v < 0)) = 0; % Clip negative values above 50% of max

% Plot the clipped voiced segment
plot(x_vclip)

% --- Autocorrelation on clipped voiced segment ---
aco = [];                    % Initialize autocorrelation array
for i = 1:500
    % Split the clipped signal into two parts: original and shifted versions
    sig1 = x_vclip(1:end-((i-1)*1));
    sig2 = x_vclip(((i-1)*1)+1:end);
    
    % Plot the two signals for comparison
    subplot(211)
    plot(sig1)
    hold on
    plot(sig2, 'r')                   % Plot the shifted version in red
    hold off
    
    % Compute autocorrelation of clipped signal
    aco(i) = sum(sig1 .* sig2);
    
    % Plot the autocorrelation result
    subplot(212)
    plot(aco)
    title('Autocorrelation of Clipped Voiced Segment')
    
    % Pause to allow visualization of each iteration
    pause(0.01)
end

% --- AMDF on clipped voiced segment ---
amdf = [];                    % Initialize AMDF array
for i = 1:500
    % Split the clipped signal into two parts: original and shifted versions
    sig1 = x_vclip(1:end-((i-1)*1));
    sig2 = x_vclip(((i-1)*1)+1:end);
    
    % Plot the two signals for comparison
    subplot(211)
    plot(sig1)
    hold on
    plot(sig2, 'r')                   % Plot the shifted version in red
    hold off
    
    % Compute AMDF of clipped signal
    amdf(i) = mean(abs(sig1 - sig2));
    
    % Plot the AMDF result
    subplot(212)
    plot(amdf)
    title('AMDF of Clipped Voiced Segment')
    
    % Pause to allow visualization of each iteration
    pause(0.01)
end