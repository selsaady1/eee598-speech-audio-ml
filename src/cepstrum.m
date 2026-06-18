clear  % Clear the workspace variables
clc    % Clear the command window
close all  % Close all open figures

% Read the audio file 'sustained_iy.wav'
[x1, fs1] = audioread('sustained_iy.wav');

% Resample the audio signal from 44.1kHz to 16kHz for further processing
x1 = resample(x1, 160, 441);  
fs1 = 16000;  % Set new sampling rate to 16 kHz

% Define the start and end points of the segment (20 ms) to analyze
nstart = 2.1e4;  % Starting sample point (based on signal length)
nend = nstart + (floor((20/1000)*fs1) - 1);  % 20 ms window based on fs1

% FFT-related parameters
Nfft = 512;  % Number of FFT points
f = linspace(0, fs1/2, Nfft/2);  % Frequency vector from 0 to Nyquist frequency

% Extract the segment of interest from the signal (20 ms duration)
x1 = x1(nstart:nend);

% Perform FFT on the selected segment of the signal
X = fft(x1, Nfft);  % Compute the FFT of the signal segment
Xlogabs = log(abs(X));  % Compute the logarithm of the magnitude spectrum
Xphase = angle(X);  % Compute the phase spectrum (not used in this demo)

% Compute the inverse FFT of the log-magnitude spectrum to get the cepstrum
C = ifft(Xlogabs);  % Cepstrum calculation, capturing periodic components

% Uncomment the following lines to plot the original signal and log magnitude spectrum
% figure
% subplot(211)
% plot(x1)
% title('Time-Domain Signal')
% subplot(212)
% plot(Xlogabs(1:Nfft/2))
% title('Log Magnitude Spectrum')

% Set the number of low quefrency (cepstral) coefficients to be modified (liftering)
Nzero = 30;  % Number of cepstral coefficients to zero out (liftering parameter)

% Choose the liftering mode (1 = high-pass liftering, 0 = low-pass liftering)
lifter = 1;  % 0 for low-pass liftering, 1 for high-pass liftering

% Plot the absolute values of the cepstral coefficients (original cepstrum)
figure(1)
plot(abs(C), 'k')  % Plot the cepstrum (before liftering)
title('Cepstrum (Before Liftering)')
xlabel('Quefrency')
ylabel('Magnitude')
hold on

% Apply liftering (modification of cepstral coefficients)
if lifter == 1
    % High-pass liftering (remove low quefrency coefficients)
    C(2:Nzero) = 0;
    C(end:-1:end-Nzero-2) = 0;
else
    % Low-pass liftering (remove high quefrency coefficients)
    C(Nzero+1:end-Nzero-2) = 0;
end

% Compute the modified log-magnitude spectrum after liftering
Xlogabs_r = real(fft(C));  % Reconstructed log magnitude spectrum after liftering

% Plot the cepstrum after liftering
plot(abs(C), 'r', 'LineWidth', 2)  % Plot modified cepstrum in red
title('Cepstrum (After Liftering)')
xlabel('Quefrency')
ylabel('Magnitude')
hold off

% Plot the original and modified log magnitude spectra for comparison
figure(2)
plot(f, Xlogabs(1:Nfft/2), 'k')  % Original log magnitude spectrum (black)
hold on
plot(f, Xlogabs_r(1:Nfft/2), 'r', 'LineWidth', 2)  % Modified log magnitude spectrum (red)
title('Log Magnitude Spectrum (Before and After Liftering)')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
legend('Original', 'After Liftering')
hold off