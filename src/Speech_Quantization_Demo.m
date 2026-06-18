% Quantization Effects on Audio Signals
%
% This script demonstrates the impact of quantizing an audio signal at different bit depths, 
% showing how reducing the resolution affects both the visual and audible quality of the signal. 
% Quantization is the process of mapping continuous amplitude values into a finite number of levels, 
% and this script compares the original high-resolution audio signal to versions quantized at lower bit depths.
%
% Key steps in this process:
% 1. **Original audio signal**:
%    - Load an audio signal ('sa1.wav') and play it at its original 16-bit resolution.
%    - Plot a small section (500 samples) of the original audio signal for visualization.
% 2. **Quantization**:
%    - Quantize the signal at 16-bit, 8-bit, 4-bit, and 2-bit resolutions.
%    - For each quantized version, overlay it on the original signal and plot the difference between them.
%    - Play each quantized version to demonstrate the perceptual degradation as the bit depth decreases.
% 3. **Fourier Transform (FT) analysis**:
%    - Use the Fourier Transform to analyze the frequency spectrum of the original and quantized signals.
%    - Compare the full signals as well as specific frames of the audio in the frequency domain.
%
% Students should observe:
% - How quantizing the audio signal at lower bit depths introduces distortion, visible in the plots and audible when played.
% - The audible degradation in the signal quality, with 2-bit quantization causing significant loss of detail.
% - The impact of quantization on the frequency content of the signal, observed through the Fourier Transform.
%
% This demonstration highlights the trade-off between bit depth and audio quality, illustrating the effects of quantization in digital audio processing.


clear
clc
close all
[x SR] = audioread('sa1.wav');
%x = 3*x;

% Plot the first 100 samples of the original signal and play it
disp('Original 16-bit version');
soundsc(x,SR)
t = [1:length(x)]/SR;              
plot(t(12500:13000),x(12500:13000))
pause 


% Quantize the signal with 16, 8, 4, and 2 bits, play it, and plot the 
% first 100 samples of the original signal with the quantized version
% overlayed
disp('16-bit quantized');
x_16bits = double(uencode(x,16))/(2^16);
x_16bits = 2*(x_16bits-mean(x_16bits));
soundsc(x_16bits,SR)
subplot(211)
plot(t(12500:13000),x(12500:13000),t(12500:13000),x_16bits(12500:13000))
subplot(212)
plot(x(12500:13000) - x_16bits(12500:13000))
pause

disp('8-bit quantized');
x_8bits = double(uencode(x,8))/(2^8);
x_8bits = 2*(x_8bits-mean(x_8bits));
soundsc(x_8bits,SR)
subplot(211)
plot(t(12500:13000),x(12500:13000),t(12500:13000),x_8bits(12500:13000))
subplot(212)
plot(x(12500:13000) - x_8bits(12500:13000))
pause

disp('4-bit quantized');
x_4bits = double(uencode(x,4))/(2^4);
x_4bits = 2*(x_4bits-mean(x_4bits));
soundsc(x_4bits,SR)
subplot(211)
plot(t(12500:13000),x(12500:13000),t(12500:13000),x_4bits(12500:13000))
subplot(212)
plot(x(12500:13000) - x_4bits(12500:13000))
pause

disp('2-bit quantized');
x_2bits = double(uencode(x,2))/(2^2);
x_2bits = 2*(x_2bits-mean(x_2bits));
soundsc(x_2bits,SR)
subplot(211)
plot(t(12500:13000),x(12500:13000),t(12500:13000),x_2bits(12500:13000))
subplot(212)
plot(x(12500:13000) - x_2bits(12500:13000))


% % FT Demo
% Plot the complete signal
plot_fft(x,SR,1)
plot_fft(x_4bits,SR,1)


% Plot a frame of the signal
plot_fft(x(12500:13000),SR,1)
plot_fft(x_2bits(12500:13000),SR,1)


