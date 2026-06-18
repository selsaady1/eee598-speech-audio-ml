clear % Clear the workspace
clc   % Clear the command window

% Load the sustained 'a' vowel audio file
[x1, fs1] = audioread('sustained_a.wav');

% Resample the audio to 16kHz since the LPC filter we will use is designed for 16kHz audio
x1 = resample(x1, 160, 441);  % Resample to 160 samples per frame
fs1 = 16000;                  % Set the new sample rate to 16kHz
fs = fs1;                     % Set the sample rate to fs1 (16kHz)

% Set frame parameters: 20 ms frame length
fr_len = 20;                    
fr_N = ((fr_len / 1000) * fs);  % Number of samples per frame (based on the frame length in ms)
shift_R = fr_N / 4;             % Frame shift (75% overlap)

% Initialize the synthesized signal as zeros (same length as original audio)
sum_w = zeros(length(x1), 1);

% 4-second mark (in samples)
frame_4_sec = 4 * fs;  

% Loop through the frames of the signal
for i = 1:shift_R:length(x1) - fr_N
    n = [i:i + fr_N - 1];           % Define the frame indices

    w = window(@triang, fr_N);      % Apply triangular window to the current frame
    xwin1 = x1(n) .* w;             % Window the current frame of the input signal
    
    [A1] = lpc(xwin1, 16);          % LPC analysis: compute LPC coefficients for the current frame (order 16)
        
    E1 = filter(A1, 1, xwin1);      % Calculate the residual (error signal) of the LPC filter

    % Reconstruct the excitation signal by summing the residuals across frames
    sum_w(n) = sum_w(n) + E1;


        % Check if the current frame is at the 4-second mark
    if i >= frame_4_sec && i < frame_4_sec + shift_R
        % Compute the log magnitude spectrum of the speech frame
        NFFT = 1024;  % FFT length
        Xwin1 = fft(xwin1, NFFT);  % FFT of the windowed speech frame
        freqs = (0:NFFT/2-1) * fs / NFFT;  % Frequency axis for plotting

        % LPC frequency response
        [h, w_lpc] = freqz(1, A1, NFFT, fs);  % Frequency response of the LPC filter
        

                % Normalize the speech spectrum to avoid any scaling issues
        Xwin1 = Xwin1 / max(abs(Xwin1));  % Normalize FFT magnitude
        h = h / max(abs(h));  % Normalize LPC frequency response to the same scale

        % Plot the log magnitude spectrum and LPC frequency response
        figure;
        plot(freqs, 20*log10(abs(Xwin1(1:NFFT/2))), 'b', 'LineWidth', 1.5); % Log magnitude spectrum of speech
        hold on;
        plot(w_lpc, 20*log10(abs(h)), 'r', 'LineWidth', 1.5); % LPC frequency response
        hold off;
        title('Log Magnitude Spectrum and LPC Frequency Response (4-second mark)');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (dB)');
        legend('Log Magnitude Spectrum', 'LPC Frequency Response');
        grid on;
        pause
    end
end

% Spectrogram parameters for visualization
fr_N = ((fr_len / 1000) * fs);      % Recalculate frame length in samples
shift_R = fr_N / 2;                 % Frame shift for spectrogram (50% overlap)
w = window(@triang, fr_N);          % Apply triangular window for spectrogram

% Compute and display the spectrogram of the original audio signal
[S, F, T, P] = spectrogram(x1, w, fr_N - shift_R, fr_N, fs);
figure
imagesc(T, F, 20 * log10(abs(S)));  % Display the magnitude of the spectrogram in decibels
axis xy                            % Set axis orientation
xlabel('Time (s)')                  % Label x-axis as time
ylabel('Frequency (Hz)')            % Label y-axis as frequency

% Compute and display the spectrogram of the reconstructed excitation signal (sum_w)
[S, F, T, P] = spectrogram(sum_w, w, fr_N - shift_R, fr_N, fs);
figure
imagesc(T, F, 20 * log10(abs(S)));  % Display the magnitude of the spectrogram in decibels
axis xy                            % Set axis orientation
xlabel('Time (s)')                  % Label x-axis as time
ylabel('Frequency (Hz)')            % Label y-axis as frequency