function plot_fft(x, Fs, ps)
% plot_fft - Plots the amplitude or power spectrum of a signal using FFT.
% 
% Syntax: plot_fft(x, Fs, ps)
%
% Inputs:
%   x  - Input signal (time-domain data).
%   Fs - Sampling frequency of the input signal (in Hz).
%   ps - Plot type:
%        1 for Power Spectrum (in dB),
%        0 for Amplitude Spectrum.
%
% Example:
%   t = 0:1/1000:1-1/1000;        % 1 second of data sampled at 1 kHz
%   x = sin(2*pi*100*t);           % A 100 Hz sine wave
%   plot_fft(x, 1000, 0);          % Plot amplitude spectrum
%   plot_fft(x, 1000, 1);          % Plot power spectrum

% Compute the number of points for FFT, equal to the length of the input signal
NFFT = length(x);

% Compute the FFT of the signal and normalize by NFFT
Y = fft(x, NFFT) / NFFT;

% Generate the frequency axis for the single-sided spectrum
% Only positive frequencies are plotted (up to Nyquist frequency)
f = Fs / 2 * linspace(0, 1, NFFT / 2 + 1);

% Create a new figure window for the plot
figure;

% Plot the power spectrum (in dB) if ps == 1
if ps == 1
    plot(f, 20*log10(abs(Y(1:floor(NFFT/2)+1))));
    title('Single-Sided Power Spectrum of Signal');
    xlabel('Frequency (Hz)');
    ylabel('Signal Power (dB)');
% Plot the amplitude spectrum if ps == 0
else
    plot(f, 2*abs(Y(1:floor(NFFT/2)+1)));
    title('Single-Sided Amplitude Spectrum of Signal');
    xlabel('Frequency (Hz)');
    ylabel('Signal Amplitude');
end
end