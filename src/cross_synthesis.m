clear % Clear the workspace
clc   % Clear the command window

% Load the audio files
[x1, fs1] = audioread('SteveJob.wav');  % Speech signal from 'SteveJob.wav'
[x2, fs2] = audioread('guitar.wav');    % Audio signal from 'guitar.wav'

% Resample the second audio to match a sampling rate of the first
if fs1 ~= fs2
    x2 = resample(x2, fs1, fs2);
    fs2=fs1;
end


% Truncate the first audio signal to the first 8 seconds
x1 = x1(1:8*fs1);

% Duplicate the second audio to match the length of the first signal
x2 = [x2; x2];

% Set the common sample rate for both signals
fs = fs1;

% Frame parameters: 20 ms frame length
fr_len = 20;                    
fr_N = ((fr_len/1000)*fs);      % Number of samples per frame
shift_R = fr_N/4;               % Frame shift (75% overlap)

% Initialize the synthesized signal
sum_w = zeros(8*fs,1);          % Preallocate space for the result

% Loop through the frames of the signals for cross-synthesis
for i = 1:shift_R:4*fs
    n = [i:i+fr_N-1];           % Define frame indices
    
    % Apply triangular window to both signals
    w = window(@triang, fr_N);  
    xwin1 = x1(n).*w;           % Windowed frame from speech signal
    xwin2 = x2(n).*w;           % Windowed frame from guitar signal
    
    % LPC analysis (Linear Predictive Coding) on the speech frame
    [A1] = lpc(xwin1,8);       % Compute LPC coefficients for speech
    E1 = filter(A1, 1, xwin1);  % Compute residual (error) signal from speech

    % LPC analysis on the guitar frame
    [A2, E2] = lpc(xwin2,8);   % Compute LPC coefficients for guitar
    E2 = filter(A2, 1, xwin2);  % Compute residual (error) signal from guitar
    
    % Gain adjustment to match the energy of the two signals
    g = sumsqr(E1) / (sumsqr(E2));
    g = sumsqr(E1) / (.3 * sumsqr(E2));
    %g=1;
    
    % Cross-synthesize by filtering the guitar residual with speech LPC coefficients
    xwin1_recon = filter(1, A1, sqrt(g) * E2);  
    
    % Accumulate the reconstructed frames into the synthesized signal
    sum_w(n) = sum_w(n) + xwin1_recon;  
end

% Plot the original and synthesized signals for comparison
subplot(311)
plot(x1(1:i))  % Plot original speech signal
title('Original Speech Signal')

subplot(312)
plot(x2(1:i))  % Plot original  signal
title('Original Signal')

subplot(313)
plot(sum_w(1:i))  % Plot the cross-synthesized signal
title('Cross-Synthesized Signal')

%soundsc(x1)  % Uncomment to listen to original speech
soundsc(sum_w)  % Play the cross-synthesized sound