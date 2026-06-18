% Quantization Effects on a Sine Wave Signal
%
% This script demonstrates the process of quantization and its impact on the quality of a signal by comparing the original signal to versions quantized at different bit depths. 
% Quantization is the process of mapping a continuous range of values into a finite range, which is commonly used in digital audio processing.
%
% Key steps in this process:
% 1. **Original signal generation**:
%    - Generate a 2-second sine wave with a frequency of 200 Hz and sample it at 3000 Hz.
%    - Plot and play the original signal.
% 2. **Quantization**:
%    - Quantize the signal at 16-bit, 8-bit, 4-bit, and 2-bit resolution.
%    - For each quantized signal, overlay the original and quantized signals in a plot.
%    - Compute and visualize the difference between the original and quantized signals.
%    - Play each quantized version to highlight the perceptual degradation as the bit depth decreases.
% 3. **Fourier Transform**:
%    - Visualize the frequency spectrum of the 4-bit quantized signal using the Fourier Transform.
%
% Students should observe:
% - How quantizing at different bit depths introduces distortion to the signal, with lower bit depths producing more noticeable errors.
% - The audible degradation in the signal quality as the bit depth decreases, demonstrating the trade-off between bit depth and signal fidelity.
% - The impact of quantization on the frequency content of the signal, particularly as the resolution decreases.
%
% This demonstration provides insight into the effects of quantization in digital signal processing, illustrating the balance between bit resolution and signal quality.

clear
clc
close all

% Step-1: specify the sampling rate
SR = 3000;                    % Sampling frequency
time_bin = 1/SR;


% Step-2: define the parameters of the sine wave
duration = 2;                      % signal length in seconds
t = 0:time_bin:duration;                % Time vector


% Signal to Plot: Sum of sinusoids
x = .8*sin(2*pi*200*t);      % 4*sin(2*pi*250*t)+ 3*sin(2*pi*350*t)+ 2*sin(2*pi*370*t)+ 5*sin(2*pi*400*t); x = x/max(x);



% Plot the first 100 samples of the original signal and play it
disp('Original 32-bit version');
soundsc(x,SR)
plot(t(1:100),x(1:100))
pause 


% Quantize the signal with 16, 8, 4, and 2 bits, play it, and plot the 
% first 100 samples of the original signal with the quantized version
% overlayed
disp('16-bit quantized');
x_16bits = double(uencode(x,16))/(2^16);
x_16bits = 2*(x_16bits-mean(x_16bits));
soundsc(x_16bits,SR)
subplot(211)
plot(t(1:100),x(1:100),t(1:100),x_16bits(1:100))
ylim([-1 1]);
subplot(212)
plot(t(1:100),x(1:100) - x_16bits(1:100))
ylim([-1 1]);
pause

disp('8-bit quantized');
x_8bits = double(uencode(x,8))/(2^8);
x_8bits = 2*(x_8bits-mean(x_8bits));
soundsc(x_8bits,SR)
subplot(211)
plot(t(1:100),x(1:100),t(1:100),x_8bits(1:100))
ylim([-1 1]);
subplot(212)
plot(t(1:100),x(1:100) - x_8bits(1:100))
ylim([-1 1]);
pause

disp('4-bit quantized');
x_4bits = double(uencode(x,4))/(2^4);
x_4bits = 2*(x_4bits-mean(x_4bits));
soundsc(x_4bits,SR)
subplot(211)
plot(t(1:100),x(1:100),t(1:100),x_4bits(1:100))
ylim([-1 1]);
subplot(212)
plot(t(1:100),x(1:100) - x_4bits(1:100))
ylim([-1 1]);
pause

disp('2-bit quantized');
x_2bits = double(uencode(x,2))/(2^2);
x_2bits = 2*(x_2bits-mean(x_2bits));
soundsc(x_2bits,SR)
subplot(211)
plot(t(1:100),x(1:100),t(1:100),x_2bits(1:100))
ylim([-1 1]);
subplot(212)
plot(t(1:100),x(1:100) - x_2bits(1:100))
ylim([-1 1]);

% % FT Demo
plot_fft(x_4bits,SR,0)
