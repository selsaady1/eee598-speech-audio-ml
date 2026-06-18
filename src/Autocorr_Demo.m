% Autocorrelation Demo
%
% This script demonstrates the concept of autocorrelation using two types of signals:
% 1. A sine wave with a frequency of 400 Hz.
% 2. Gaussian white noise (random signal).
%
% Autocorrelation measures how well a signal matches a time-shifted version of itself.
% In the first part of the demo, you will see how a periodic signal (sine wave) shows a
% clear repetitive pattern in its autocorrelation.
% In the second part, the autocorrelation of a random noise signal is shown,
% which tends to decay more rapidly and exhibits less predictable patterns.
%
% The script visualizes the two parts of the signal being correlated and plots
% the resulting autocorrelation, which updates dynamically as the signals slide over each other.
%
% You should observe the following:
% - For the sine wave, the autocorrelation shows periodic peaks corresponding to the periodic nature of the signal.
% - For random noise, the autocorrelation quickly diminishes, indicating the lack of periodicity in the signal.
%
% Additionally, you can hear the sound of the sine wave at the start of the demo.
% You can also uncomment the line to hear the random noise.

% Clear workspace and command window
clear
clc

% Define sampling frequency and signal length
fs = 5000;       % Sampling frequency (5000 Hz)
len = 3;         % Signal length (3 seconds)

% Create a time vector (in samples)
t = [1:len*fs];

% Generate a sine wave signal with a frequency of 400 Hz and amplitude of 0.5
x = 0.5 * sin(2*pi*(400/fs)*t);

% Take the first 200 samples of the sine wave for autocorrelation calculation
x_v = x(1:200);

% Play the sine wave signal
sound(x, fs)

% Initialize autocorrelation array
aco = [];

% Loop to compute the autocorrelation of the signal
for i = 1:100
    % Divide the signal into two parts (sliding one over the other)
    sig1 = x_v(1:end-((i-1)*1));        % First part of the signal
    sig2 = x_v(((i-1)*1)+1:end);        % Second part, shifted by i samples
    
    % Plot the two signals for visualization
    subplot(211)
    plot(sig1)                          % Plot first part
    hold on
    plot(sig2, 'r')                     % Plot second part in red
    hold off
    
    % Compute the autocorrelation (sum of pointwise multiplication)
    aco(i) = sum(sig1 .* sig2);
    
    % Plot the autocorrelation result
    subplot(212)
    plot(aco)
    title('Autocorrelation of Sine Wave')
    
    % Pause to allow for animation effect
    pause(0.01)
end

pause;

% Generate random noise for the second part of the demo
x_v = randn(200,1);     % Generate 200 samples of Gaussian white noise

% Uncomment the following line if you want to hear the random noise
% sound(x, fs)

% Initialize autocorrelation array for the noise signal
aco = [];

% Loop to compute autocorrelation for the random noise signal
for i = 1:100
    % Divide the noise signal into two parts (sliding one over the other)
    sig1 = x_v(1:end-((i-1)*1));        % First part of the noise signal
    sig2 = x_v(((i-1)*1)+1:end);        % Second part, shifted by i samples
    
    % Plot the two signals for visualization
    subplot(211)
    plot(sig1)                          % Plot first part of noise signal
    hold on
    plot(sig2, 'r')                     % Plot second part in red
    hold off
    
    % Compute the autocorrelation of the noise signal
    aco(i) = sum(sig1 .* sig2);
    
    % Plot the autocorrelation result for the noise
    subplot(212)
    plot(aco)
    title('Autocorrelation of Noise')
    
    % Pause to allow for animation effect
    pause(0.05)
end