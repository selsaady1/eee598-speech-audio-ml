% FIR Filtering and LTI System Properties Demo
%
% This script demonstrates the application of FIR (Finite Impulse Response) filters 
% and explores the fundamental properties of Linear Time-Invariant (LTI) systems.
% It uses an averaging filter to smooth an audio signal and visualizes the effects 
% of time-shifting, superposition, and scaling, which are key properties of LTI systems.
%
% Key steps in this process:
% 1. **FIR filtering**: A simple averaging filter is applied to an audio signal to smooth it.
% 2. **Energy comparison**: The energy of the original and filtered signals is calculated, and the energy ratio is computed.
% 3. **LTI system properties**:
%    - **Time-shifting**: Demonstrating that filtering a time-shifted signal is equivalent to shifting the filtered signal.
%    - **Superposition**: Filtering two signals individually vs. filtering their sum.
%    - **Scaling**: Showing the linearity of scaling both the signal and the filtered result.
% 4. **Filter order**: Illustrating the effect of applying two filters in different orders and comparing their outputs.
% 5. **Frequency response**: Visualizing the frequency response of the applied filters (averaging and high-pass).
%
% Students should observe:
% - How the FIR filter smooths the audio signal.
% - The key properties of LTI systems, such as the consistency of filtering with time-shifting, superposition, and scaling.
% - The impact of applying filters in different orders.
% - The frequency response of the filters, including how the high-pass filter suppresses low frequencies.
%
% This demonstration provides insights into the behavior of LTI systems and how FIR filters alter signals both in the time domain and frequency domain.


% Clear the workspace and command window
clear
clc

% Load audio file 'sa2.wav'
[x, fs] = audioread('sa2.wav');

% Get the length of the audio signal
len_x = length(x);

% Create time vector for the audio signal
t = [1:len_x]/fs;

% Plot the original audio signal in the time domain
figure
plot(t,x)
title('Time-Domain Audio Signal')
xlabel('Time (seconds)')
ylabel('Amplitude')

% Play the original audio signal
soundsc(x, fs)

pause;

% --- Averaging FIR filtering ---

% Define filter length (M-point averaging filter)
M = 15;                    % Length 15 averaging filter
b = (1/M)*ones(M,1);        % Coefficients of the averaging filter

% Apply the filter to the audio signal (smoothing it)
y = conv(b, x);             % Convolution for filtering (can also use filter function)

% Play the filtered audio signal
soundsc(y, fs)              % soundsc rescales the signal to avoid clipping

% Calculate the energy of the original and filtered signals
E_x = sqrt(sumsqr(x));      % Energy of original signal
E_y = sqrt(sumsqr(y));      % Energy of filtered signal

% Calculate the ratio of energies (filtered/original)
E_ratio = E_y / E_x;

%%% Linear Time Invariance (LTI) System Properties %%%

% --- Time Shift ---

% Add one second of silence (zeros) at the beginning of the signal
x_shifted = [zeros(fs,1); x];    % Shift signal by 1 second

% Apply the same averaging filter to the shifted signal
y_shifted = conv(b, x_shifted);

% Plot the original filtered signal, the shifted filtered signal, and their difference
figure
subplot(311)
plot([zeros(fs,1); y])           % Original filtered signal with one second of zeros
title('Filtered Signal with Zero Padding (Shifted)')
subplot(312)
plot(y_shifted)                  % Shifted filtered signal
title('Shifted Filtered Signal')
subplot(313)
plot(y_shifted - [zeros(fs,1); y]) % Difference between the two signals
title('Difference Between Shifted and Non-Shifted Filtered Signals')

% --- Superposition ---

% Load a second audio signal ('sa1.wav')
[x2, fs2] = audioread('sa1.wav');

% Apply the averaging filter to both signals
y2 = conv(b, x2(1:2*fs));        % Filter first 2 seconds of x2
y = conv(b, x(1:2*fs));          % Filter first 2 seconds of x

% Add the two signals together (superposition)
x_sp = x(1:2*fs) + x2(1:2*fs);

% Apply the averaging filter to the superposed signal
y_sp = conv(b, x_sp);

% Plot the sum of filtered signals, the filtered sum, and their difference
figure
subplot(311)
plot(y2 + y)                     % Sum of filtered signals
title('Sum of Filtered Signals')
subplot(312)
plot(y_sp)                       % Filtered superposed signal
title('Filtered Superposed Signal')
subplot(313)
plot(y_sp - (y2 + y))            % Difference between the two
title('Difference Between Superposition and Filtered Sum')

% --- Scaling ---

% Scale the original signal by 1.5
x_scal = 1.5 * x;

% Apply the filter to the scaled signal
y_scal = conv(b, x_scal);

% Apply the filter to the original signal (for comparison)
y = conv(b, x);

% Plot scaled filtered signal, 1.5 * original filtered signal, and their difference
figure
subplot(311)
plot(1.5 * y)                    % Scaled version of the filtered signal
title('1.5x Scaled Filtered Signal')
subplot(312)
plot(y_scal)                     % Filtered scaled signal
title('Filtered Scaled Signal')
subplot(313)
plot(y_scal - 1.5 * y)           % Difference between the two
title('Difference Between Scaled and Filtered Signals')

% --- Changing the Order of LTI Systems ---

% Define two different averaging filters
M1 = 9;                          % Length 9 averaging filter
b1 = (1/M1)*ones(M1,1);

M2 = 3;                          % Length 3 averaging filter
b2 = (1/3)*ones(3,1);

% Apply the filters in two different orders
y12 = conv(b2, conv(b1, x));     % Apply b1, then b2
y21 = conv(b1, conv(b2, x));     % Apply b2, then b1

% Compute the combined filter response
y1122 = conv(conv(b1, b2), x);   % Equivalent to convolving b1 and b2 together first

% Plot the results of different filter orders and their differences
figure
subplot(411)
plot(y12)
title('Filtered Signal (b1 then b2)')
subplot(412)
plot(y21)
title('Filtered Signal (b2 then b1)')
subplot(413)
plot(y12 - y21)
title('Difference Between Filter Orders')
subplot(414)
plot(y12 - y1122)
title('Difference Between Sequential and Combined Filter Applications')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Frequency Response of the Filter ---

% Define the frequency axis (normalized from 0 to pi radians)
w = linspace(0, pi, 1024);

% Compute the frequency response of the averaging filter
H = freqz(b, 1, w);

% Plot the magnitude of the frequency response
figure
subplot(211)
plot(w, abs(H))
title('Frequency Response of Averaging Filter (Magnitude)')
xlabel('Normalized Frequency (radians)')
ylabel('Magnitude')

% Plot the magnitude in decibels (dB)
subplot(212)
plot(w, 20*log10(abs(H)))
title('Frequency Response of Averaging Filter (Magnitude in dB)')
xlabel('Normalized Frequency (radians)')
ylabel('Magnitude (dB)')

% --- High-Pass Filter ---

% Design a high-pass FIR filter with cutoff frequency of (1/8)*Nyquist
% frequency (or half the sampling rate)
b = fir1(150, 1/8, 'high');

% Compute and plot the frequency response of the high-pass filter
H = freqz(b, 1, w);

figure
subplot(211)
plot(w, abs(H))
title('Frequency Response of High-Pass Filter (Magnitude)')
xlabel('Normalized Frequency (radians)')
ylabel('Magnitude')

subplot(212)
plot(w, 20*log10(abs(H)))
title('Frequency Response of High-Pass Filter (Magnitude in dB)')
xlabel('Normalized Frequency (radians)')
ylabel('Magnitude (dB)')