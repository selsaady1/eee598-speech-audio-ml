%% EEE598 – KWS Project: Part 1 - Preprocessing
% Section Summary:
% This first section prepares the raw Google Speech Commands audio clips for training.
% It performs the following:
%   1. Reads the keyword folders already in the current directory (yes, no, stop, go)
%   2. Resamples all audio to 16 kHz for consistency
%   3. Removes leading and trailing silence using short-time energy
%   4. Normalizes amplitude so all clips are ona a similar loudness scale
%   5. Pads or truncates each clip to exactly 1.0 second in length
%   6. Splits the processed data into training, validation, and test sets
%   7. Saves everything into a new folder called “Preprocessed”
%
% This ensures that all audio clips have the same format, sample rate,
% and duration before feature extraction and model training.

% clear; clc; close all;

% ----- Basic setup -----
keywords = {'yes','no','stop','go'};   % Target keywords to include in the analysis
targetFs = 16000;                    % Target sampling frequency for resampling
clipLen  = 1.0;                      % Final duration (seconds) for each audio clip
outRoot  = fullfile(pwd,'Preprocessed'); % Folder where theprocessed data will be saved

% Creating the "Preprocessed" folder (assuming it doesn't already exist)
if ~exist(outRoot,'dir') 
    mkdir(outRoot); 
end

% ----- Parameters for silence trimming -----
frame_ms = 20;         % Frame length for energy calculation (ms)
hop_ms = 10;         % Hop size between frames (ms)
winFunc = @hamming;    % Window function used for energy analysis
energyFloor_dB = -45;   % Energy threshold (in dB) to detect silence regions
minKeep_ms = 120;      % Minimum amount of audio to keep (avoids cutting out short sounds)

% ----- Dataset splitting parameters -----
splitRatio = [0.8 0.1 0.1];   % Splitting the data (80% train, 10% validation, 10% test)
rng(598);                     % Set random seed for reproducibility

% ----- Looping over keyword folders -----
for k = 1:length(keywords)
    word = keywords{k};                         % Getting the current keyword (ex. 'yes')
    files = dir(fullfile(word,'*.wav'));         % List all .wav files in that keyword folder
    fprintf('\nProcessing class "%s" (%d files)\n',word,length(files));   % Display progress in console

     % ----- Randomly shuffle file order and split indices -----
    N = length(files);             % Total number of files for this word
    idx = randperm(N);              % Randomizes file order
    nTr = round(splitRatio(1)*N);    % Number of training files (80%)
    nVa = round(splitRatio(2)*N);    % Number of validation files (10%)
    
    % Storing index ranges for each dataset split
    sets = { {'train', idx(1:nTr)}, ...                 % Training set indices
             {'val',   idx(nTr+1:nTr+nVa)}, ...         % Validation set indices
             {'test',  idx(nTr+nVa+1:end)} };            % Test set indices

% ----- Create output subfolders (train/val/test) -----
    for s = 1:length(sets)
        splitName = sets{s}{1};                         % Get the name of the current split
        outDir = fullfile(outRoot, splitName, word);     % Path: Preprocessed/train/yes, etc.
        if ~exist(outDir,'dir'),                        % Creates folder if it doesn’t exist yet
            mkdir(outDir); 
        end       
    end

    % ----- Processing each .wav file for current keyword -----
    for i = 1:N
        [x, fs] = audioread(fullfile(word,files(i).name));   % Read audio signal and sample rate
        if size(x,2)>1, x = mean(x,2); end          % (to mono)
        if fs ~= targetFs, x = resample(x,targetFs,fs); end   % Resample if sampling rate differs from target (16 kHz)
        x = x ./ max(abs(x)+eps);                    % Normalize waveform amplitude to range [-1, 1]

        % ----- Trims silence using short-time energy -----
        % Removes low-energy regions (before/after speech)
        % Uses the window, hop, and threshold values defined earlier
        x = trim_by_energy(x, targetFs, frame_ms, hop_ms, ...
                           winFunc, energyFloor_dB, minKeep_ms);

        % --- Pads or crops to exactly 1 s ---
        Ntarget = round(targetFs*clipLen);  % Desired sample length = 1s × 16kHz
        x = fix_length(x, Ntarget);          % Pad with zeros or truncate extra samples

        % determine which split this file belongs to
        if ismember(i, sets{1}{2}), splitName='train';
        elseif ismember(i, sets{2}{2}), splitName='val';
        else, splitName='test';
        end

        % ----- Saves the preprocessed waveform -----
        % Store as a .mat file containing 'x' (waveform) and 'targetFs' (sample rate)
        outDir = fullfile(outRoot, splitName, word);
        [~,base,~] = fileparts(files(i).name);
        save(fullfile(outDir,[base '.mat']), 'x','targetFs');
    end
end








% ----- Building Extra "silence" class from _background_noise_ -----
% We analyze 1-second chunks of the Google background noise recordings
% as examples of a silence / non-keyword class.

bgdir = fullfile(pwd,'_background_noise_');  % folder that contains doing_the_dishes.wav, etc.

if ~exist(bgdir,'dir')
    warning('Silence class: folder "_background_noise_" not found. Skipping silence class.');
else
    % Load all background noise recordings, convert to mono, resample to targetFs
    bgSignals = {};
    dd = dir(fullfile(bgdir,'*.wav'));
    for i = 1:numel(dd)
        [nb, fsb] = audioread(fullfile(dd(i).folder, dd(i).name));
        if size(nb,2) > 1, nb = mean(nb,2); end      % mono
        if fsb ~= targetFs, nb = resample(nb,targetFs,fsb); end
        bgSignals{end+1} = nb(:); 
    end

    % How many "silence" clips to generate total
    Nsil = 2000;      % can adjstu (1500–3000)

    % Random train/val/test split for silence, using same splitRatio
    idx = randperm(Nsil);
    nTr = round(splitRatio(1)*Nsil);
    nVa = round(splitRatio(2)*Nsil);
    sets_sil = { {'train', idx(1:nTr)}, ...
                 {'val',   idx(nTr+1:nTr+nVa)}, ...
                 {'test',  idx(nTr+nVa+1:end)} };

    % Create output folders for "silence"
    for s = 1:numel(sets_sil)
        splitName = sets_sil{s}{1};
        outDir = fullfile(outRoot, splitName, 'silence');
        if ~exist(outDir,'dir'), mkdir(outDir); end
    end

    % Generate random 1-second chunks from the background noises
    Ntarget = round(targetFs * clipLen);  % number of samples for 1.0 s

    for i = 1:Nsil
        % Picks a random background recording
        nb = bgSignals{randi(numel(bgSignals))};

        % Ensures it's at least 1 second long 
        if numel(nb) < Ntarget
            nb = repmat(nb, ceil(Ntarget/numel(nb)), 1);
        end

        % Random starting point for 1-second chunk
        startIdx = randi(numel(nb) - Ntarget + 1);
        x = nb(startIdx:startIdx+Ntarget-1);

        % Normalize amplitude
        x = x ./ max(abs(x) + eps);

        % makingsure length is exactly 1 second
        x = fix_length(x, Ntarget);

        % Deciding which split this sample belongs to
        if ismember(i, sets_sil{1}{2})
            splitName = 'train';
        elseif ismember(i, sets_sil{2}{2})
            splitName = 'val';
        else
            splitName = 'test';
        end

        % Save waveform (.mat), just like the keywords
        outDir = fullfile(outRoot, splitName, 'silence');
        base = sprintf('bg_%04d', i);
        save(fullfile(outDir, [base '.mat']), 'x', 'targetFs');
    end

    fprintf('Built %d "silence" clips from _background_noise_.\n', Nsil);
end



% Userfriendly way tio indicate completion of the preprocessing step
disp('Preprocessing finished! Data saved under "Preprocessed/".');







% ----- Relevant functions -----
% These functions are used by the preprocessing loop above to:
%   1. Remove silence based on short-time energy
%   2. Center audio after trimming
%   3. Pad or crop each clip to exactly 1 second
%   These functions to help script with versatility towards ANY clip, not
%   just google ones it is trained/tested on,
%   ensuring data cleanliness, consistency, and reusability/future-proofign
% -------------------------------------------------------------------------------
% Function: trim_by_energy
% Purpose:  Removes silence from the beginning and end of the audio signal
%           using short-time energy thresholding.
function xout = trim_by_energy(x, fs, fr_ms, hop_ms, winFunc, floor_dB, minKeep_ms)
N = round(fr_ms/1000 * fs);    % Frame length in samples
H = round(hop_ms/1000 * fs);    % Hop length in samples
w = winFunc(N);                % Window (Hamming)
E = [];                         % Initializes array to hold frame energies

% Step 1: Computes short-time energy across frames 
for n=1:H:(length(x)-N)
    fr = x(n:n+N-1).*w;
    E(end+1)=sum(fr.^2); 
end

% Step 2: Finds which frames are above the silence threshold 
E = E(:); Emax=max(E); 
Eth=max(Emax*10^(floor_dB/10),eps);
idx=find(E>Eth);     % Keeps frames above threshold

% Step 3: If no voiced region is found, keep at least a minimum amount
if isempty(idx)
    keepN=round(minKeep_ms/1000*fs);    % Minimum samples to keep
    xout=take_centered(x,keepN);     % Takes a centered portion
    return
end
% --- Step 4: Find first and last "active" frames and extend slightly
first = (idx(1)-1)*H+1; last=min((idx(end)-1)*H+N,length(x));
keepN = max(last-first+1, round(minKeep_ms/1000*fs));  % Ensures minimum length

% Step 5: Output trimmed and centered audio segment
xout = take_centered(x(first:last),keepN); 
end

% --------------------------------------------------------------------------------

% Function: take_centered
% Purpose:  Centers a shorter audio segment within a larger fixed length.
%           Used to pad or crop so the signal remains centered.
% Inputs: x - input audio, N - desired total length (samples)
% Outputs: y - audio padded or cropped to length N
function y = take_centered(x,N)    
if length(x)>=N                    
    % If audoi is longer than target, crop from the middle
    extra=length(x)-N; s0=floor(extra/2)+1;
    y=x(s0:s0+N-1);
else
    % If audoi is shorter, pad equally on both sides with zeros
    pad=N-length(x); 
    lp=floor(pad/2);   % Left padding
    rp=pad-lp;         % Right padding
    y=[zeros(lp,1);
    x(:);zeros(rp,1)];
end
end

% ----------------------------------------------------------------------------------
% Function: fix_length
% Purpose:  Ensures that the output clip is exactly N samples long (1 second)
%           by padding zeros or cropping extra samples if needed.
% Inputs: x - input audio, N - desired length (samples)
% Output: y - output audio of exact length N

function y=fix_length(x,N)
if length(x)== N
    y = x;           % Already the correct length
elseif length(x)>N
    y=x(1:N);         % Crops extra samples
else
    y=[x;zeros(N-length(x),1)];      % Pad with zeros to reach target length
end
end




%% KWS Part 2: Feature Extraction (Energy, ZCR, MFCCs)
% Section Summary:
% Reads the preprocessed 1.0 s clips saved earlier:
%   Preprocessed/{train,val,test}/{class}/*.mat  (each .mat holds x, targetFs)

% For each clip, we compute per-frame features (20 ms frames, 10 ms hop):
%   - Short-Time Energy
%   - Zero-Crossing Rate (ZCR)
%   - MFCCs (STFT -> Mel filterbank -> log -> DCT, keep c0..c12)

% Then we summarize each stream at the clip level using [mean, std] across time.
%   Final clip feature vector = [E_mean, E_std, Z_mean, Z_std,
%                                MFCC_means(13), MFCC_stds(13)]  --> 30 dimensiotns.
%
% Output: features_{split}.mat   with:
%   X : [N_clips x D]  feature matrix
%   y : [N_clips x 1]  numeric labels (1=’yes’, 2=’no’, 3=’stop’, 4=’go’)
%   meta: struct with file paths, class names, and the split name

% clear; clc; close all;

% ---------------- Config (class-style params) ----------------
inRoot   = fullfile(pwd,'Preprocessed');   % input from previous step
splits   = {'train','val','test'};
classes  = {'yes','no','stop','go','silence'};       % classes we want to identify
% Frame/feature parameters:
fr_ms    = 20;                             % frame length (ms)
hop_ms   = 10;                             % frame hop (ms)
winFunc  = @hamming;                       % window
numMels  = 26;                             % Mel filterbank size
numCeps  = 13;                             % # MFCC kept (c0..c12)
nfft     = 512;                            % FFT size
eps_floor= 1e-10;                          % numerical floor for logs

% --------------- Iterate splits and build features ---------------
for s = 1:numel(splits)
    split = splits{s};
    fprintf('\n=== Split: %s ===\n', split);

    % Collecting all files & labels for this split
    fileList = {};         % cell array of full paths to .mat clips
    y = [];                % numeric labels aligned with filelist
    for ci = 1:numel(classes)
        cls = classes{ci};        % class name (string)
        d = dir(fullfile(inRoot, split, cls, '*.mat'));     % all clips for this class/split
        for k = 1:numel(d)
            fileList{end+1,1} = fullfile(d(k).folder, d(k).name); 
            y(end+1,1) = ci;     % numeric label = class index
        end
    end

    % Prepareing containers to hold per-clip feature vectors
    Xcell = cell(numel(fileList),1);   % one row (vector) per clip
    meta  = struct('files',{fileList}, 'classes',{classes}, 'split',split);

    % ----- Process each file in this split -----

    for i = 1:numel(fileList)

        % Loading preprocessed clip (.mat file) which contains: x  = waveform, targetFs = sample rate (16 kHz)
        data = load(fileList{i}); % expects variables: x, targetFs
        x  = data.x; 
        Fs = data.targetFs;

        % Frame & window setup
        N = round(fr_ms/1000 * Fs);   % Frame length in samples (20 ms)
        H = round(hop_ms/1000 * Fs);  % Hop size in samples (10 ms)
        w = winFunc(N);               % Hamming window applied to each frame

        x = x(:);     % Ensures the waveform is a column vector (for consistency)

        % Computing time-domain features (Energy & ZCR per frame)
        % feat_energy_zcr() returns:
         %  E : vector of short-time energy values (1 per frame)
        %   Z : vector of zero-crossing counts (1 per frame)
        [E, Z] = feat_energy_zcr(x, N, H, w);

        % Computing MFCCs per frame (manual STFT + Mel + DCT)
        % Returns MF = [numCeps x numFrames]
        MF = feat_mfcc_frames(x, Fs, N, H, w, nfft, numMels, numCeps, eps_floor);

        % ----- Aggregate all per-frame values to a single feature vector 
        % Each clip becomes one row vector combining: 
        % mean(E), std(E), mean(Z), std(Z), mean and std of each MFCC coefficient
        featVec = [ mean(E), std(E), ...
                    mean(Z), std(Z), ...
                    mean(MF,2).', std(MF,0,2).' ];  % MF: (numCeps x nFrames)

        Xcell{i} = featVec;    % Storesthis clip’s feature vector in the cell array
    end

    % Converting cell array to matrix 
    % Each row = one clip, each column = one feature dimension (30)

    X = cell2mat(Xcell);

    % Saving this split’s features
    outFile = fullfile(pwd, sprintf('features_%s.mat', split));
    save(outFile, 'X','y','meta');                             % save features, labels, and metadata
    fprintf('Saved %s  ->  X: %d x %d, y: %d\n', outFile, size(X,1), size(X,2), numel(y));
end

% Indicaiton of its completion as visual 
disp(' Feature extraction complete. Produced features_train/val/test.mat');


% -------------- Relevant FUncitons (for section 2) -----------------------

function [E, Z] = feat_energy_zcr(x, N, H, w)
% Short-time Energy & ZCR per frame
E = []; Z = [];
for n = 1:H:(length(x)-N+1)
    fr = x(n:n+N-1);
    frw = fr .* w;
    E(end+1) = sum(frw.^2); 

    % ZCR 
    Z(end+1) = sum( fr(1:end-1).*fr(2:end) < 0 );
end
E = E(:); Z = Z(:);
% normalize each stream (helpful for scaling)
if ~isempty(E), E = E / max(E + eps); end
if ~isempty(Z), Z = Z / max(Z + eps); end
end

function MF = feat_mfcc_frames(x, Fs, N, H, w, nfft, numMels, numCeps, eps_floor)
% Returns MFCCs as numCeps x nFrames (c0..c{numCeps-1})
% Process: STFT magnitude -> power -> Mel filterbank -> log -> DCT

% Precompute Mel filterbank (once per call)
fbank = mel_filterbank(numMels, nfft, Fs, 0, Fs/2); % (numMels x (nfft/2+1))
dctM  = dctmtx(numMels);                            % DCT matrix
dctM  = dctM(1:numCeps, :);                         % keep first numCeps rows

MFframes = [];
for n = 1:H:(length(x)-N+1)
    fr = x(n:n+N-1) .* w;
    Xk = fft(fr, nfft);
    Xk = Xk(1:(nfft/2+1));             % single-sided
    Pk = (abs(Xk).^2);                % power spectrum

    % Apply Mel filterbank
    melE = fbank * Pk(:);             % (numMels x 1)
    melE = max(melE, eps_floor);         % avoid log(0)
    logMel = log(melE);

    % DCT -> MFCCs
    c = dctM * logMel;                 % (numCeps x 1)
    MFframes(:, end+1) = c; 
end
MF = MFframes; % (numCeps x nFrames)
end

function fb = mel_filterbank(numMels, nfft, Fs, fmin, fmax)
% Triangular Mel filterbank
% Output: (numMels x (nfft/2+1))
if nargin < 4, fmin = 0; end
if nargin < 5, fmax = Fs/2; end

% Hertz <-> Mel
hz2mel = @(f) 2595*log10(1 + f/700);
mel2hz = @(m) 700*(10.^(m/2595) - 1);

% Mel points (includes 2 extra for edges)
mels = linspace(hz2mel(fmin), hz2mel(fmax), numMels + 2);
hz   = mel2hz(mels);
bins = floor((nfft+1) * hz / Fs);

fb = zeros(numMels, nfft/2+1);
for m = 1:numMels
    b0 = bins(m);   b1 = bins(m+1);  b2 = bins(m+2);
    b0 = max(b0,1); b2 = min(b2, nfft/2+1);

    % Rising slope
    for k = b0:b1
        fb(m,k) = (k - b0) / max(b1 - b0, 1);
    end
    % Falling slope
    for k = b1:b2
        fb(m,k) = (b2 - k) / max(b2 - b1, 1);
    end
end
end


%% KWS Part 3 — Training & Evaluation (k-NN / SVM)
% Section Summary:
% Uses the feature matrices created earlier (.mat):
%   features_train.mat, features_val.mat, features_test.mat

% Compares two “views” of the same data:
%   (1) MFCC + Energy + ZCR  (all 30 features)
%   (2) MFCC-only            (drops the first 4 time-domain features)

% For each view, it:
%   - Standardizes features using training set statistics
%   - Trains and validates both k-NN and SVM classifiers
%   - Reports accuracy and produces confusion matrices

% clear; clc; close all;

% ---------------- Loadinf features ----------------
% training features, validation features, test features; Extracting matrices and label vectors
S_tr = load('features_train.mat');  Xtr = S_tr.X;  ytr = double(S_tr.y);  meta = S_tr.meta;
S_va = load('features_val.mat');    Xva = S_va.X;  yva = double(S_va.y);
S_te = load('features_test.mat');   Xte = S_te.X;  yte = double(S_te.y);

classes = meta.classes;  % cell array of class names (ex. {'yes','no','stop','go', 'silence})
fprintf('Loaded: train=%d, val=%d, test=%d, dims=%d\n',size(Xtr,1),size(Xva,1),size(Xte,1),size(Xtr,2));

% ------------- Feature views -------------
% Our extractor produced 30 total features per clip:
% [E_mean, E_std, Z_mean, Z_std, MFCC_means(13), MFCC_stds(13)]
ix_full  = 1:size(Xtr,2);                  % MFCC + Energy + ZCR  (all) (1 to 30)
ix_mfcc  = 5:size(Xtr,2);                  % MFCC-only (drops first 4 dims: E/ZCR) (MFCC only)

% Defining which “views” (feature sets) we’ll evaluate
views = { struct('name','MFCC+Energy+ZCR','ix',ix_full), ...
          struct('name','MFCC-only','ix',ix_mfcc) };

% ------------- Small helper funcs -------------
zscore_fit  = @(X) deal(mean(X,1), std(X,[],1));     % Compute z-score normalization statistics
zscore_apply= @(X,mu,sig) (X - mu)./max(sig,1e-12);   % Apply z-score normalization (subtract mean, divide by std)

acc = @(yt, yp) mean(yt(:)==yp(:));    % Simple accuracy function (fraction correct/tot)

% ----------- Train & validate per feature view -------------
rng(598);
for v = 1:numel(views)
    name = views{v}.name;   % view name for printing
    ix = views{v}.ix;    % feature indices to use for this view

    % ----- standardize (train stats only) -----
    [mu, sg] = zscore_fit(Xtr(:,ix));       % computes mean & std from train set
    Xtr_s = zscore_apply(Xtr(:,ix), mu, sg);    % standardize train
    Xva_s = zscore_apply(Xva(:,ix), mu, sg);    % standardize val
    Xte_s = zscore_apply(Xte(:,ix), mu, sg);  % standardize test

    fprintf('\n=== View: %s ===\n', name);

    % _______________ k-NN (validate) ___________
    Kgrid = [1 3 5 7];       % small range of neighbor counts to try
    best.knn.acc = -inf;     % tracks best accuracy
    best.knn.K = NaN; 
    best.knn.mdl = [];

    % Train k-NN model (fitcknn func automatically stores training data)
    for K = Kgrid
        mdl = fitcknn(Xtr_s, ytr, 'NumNeighbors',K, 'Standardize',false);
        yv  = predict(mdl, Xva_s);     % Predict on validation set
        a   = acc(yva,yv);           % Compute validation accuracy
        if a > best.knn.acc, 
            best.knn = struct('acc',a,'K',K,'mdl',mdl);          % Keeps model if it performs best so far
        end
    end
    fprintf('k-NN val-best: K=%d, acc=%.2f%%\n', best.knn.K, 100*best.knn.acc);

% ================== SVM via ECOC (validate) ===================
% We testign several combinations of KernelScale and BoxConstraint
% and pick the combination that gives the highest validation accuracy.

scales = [0.5 1 2];           % Candidate kernel width values
boxes  = [0.5 1 2 4];       % Candidate C (box constraint) values
best.svm.acc = -inf;          % Track best validation accuracy
best.svm.params = [];         % Store best (scale, C)
best.svm.mdl = [];           % Store best trained model

% Trying all parameter combinations
for ks = scales
    for C = boxes
        % Defining the SVM template matlab function with current parameters
        t = templateSVM('KernelFunction','rbf', ...    % RBF = Gaussian kernel
                        'KernelScale', ks, ...            % controls width of the Gaussian
                        'BoxConstraint', C, ...       % penalty for misclassification
                        'Standardize', false);         % already standardized earlier
        % Trains ECOC (multi-class) model on training set
        mdl = fitcecoc(Xtr_s, ytr, ...
                       'Learners', t, ...
                       'Coding', 'onevsone', ...     % one vs. one SVM scheme is used
                       'ClassNames', unique(ytr));

    
        yv  = predict(mdl, Xva_s);     % Predicting validation labels
        a   = mean(yva == yv);       % Compute accuracy on validation set

        % Keeping model if it’s the best so far
        if a > best.svm.acc
            best.svm = struct('acc',a,'params',[ks C],'mdl',mdl);
        end
    end
end

% Display the best SVM hyperparameters and accuracy founds
fprintf('SVM(ECOC) val-best: KernelScale=%.2g, C=%.2g, acc=%.2f%%\n', ...
        best.svm.params(1), best.svm.params(2), 100*best.svm.acc);

    % ----- Comparison of SVM vs k-NN on validation (part of our project deliverables/analysis) 
    % Whichever performs better on validation becomes the chosen model family
    if best.svm.acc >= best.knn.acc
        fam = 'SVM';
    else
        fam = 'kNN';
    end
    fprintf('Selected family on val: %s\n', fam);

    % _________________ Retrain on train+val ___________________
    % Now it will merge the training and validation sets and retrain the chosen model
    % This gives the model access to all tehavailable labeled data (except test set)
    Xtv = [Xtr_s; Xva_s];     % Combined features
    ytv = [ytr; yva];      % Combined labels
 
    switch fam
        % ---- Retrain k-NN ----
        case 'kNN'
            mdl_final = fitcknn(Xtv, ytv, 'NumNeighbors',best.knn.K, 'Standardize',false);
        
        % ---- Retrain SVM ----
        case 'SVM'
            ks = best.svm.params(1); C = best.svm.params(2);
            t  = templateSVM('KernelFunction','rbf', ...
                     'KernelScale', ks, ...
                     'BoxConstraint', C, ...
                     'Standardize', false);
            mdl_final = fitcecoc([Xtr_s; Xva_s], [ytr; yva], ...
                         'Learners', t, ...
                         'Coding', 'onevsone', ...
                         'ClassNames', unique([ytr; yva]));
    end

    % __________________ Evaluate on test ______________
    % After the model selection and retraining, now evaluating the final chosenmodel
     % (either SVM or k-NN) on the unseen test set to estimate true performance
    yhat = predict(mdl_final, Xte_s);     % Predicts labels for all test clips
    A    = acc(yte, yhat);          % Compute overall test accuracy for test set
    fprintf('TEST accuracy (%s): %.2f%%\n', name, 100*A);

    % Save model & scalers if want to deploy later
    % Contains everything MATLAB needs to recognize future speech clips without retraining 
    % needed for part 4 below
    save(sprintf('model_%s.mat', regexprep(name,'[^a-zA-Z0-9]','')), ...
         'mdl_final','mu','sg','ix','classes');











%--------------------   
% Creating and displayign confusion matrix for current feature view
% This givess us visualization of which keywords were correctly/incorrectly predicted
% (Rows = true classes; columns = predicted classes)
% Row normalization shows results as percentages per true word.
    yte_c  = categorical(yte,  1:numel(classes), classes);    % True labels
    yhat_c = categorical(yhat, 1:numel(classes), classes);     % Predicted labels
     
    figure('Name', sprintf('Confusion — %s', name));
    confusionchart(yte_c, yhat_c, ...
        'RowSummary','row-normalized', ...
        'ColumnSummary','column-normalized');
    title(sprintf('%s (Acc=%.1f%%)', name, 100*A));
    xlabel('Predicted'); ylabel('True');

end


disp('Training & evaluation complete.');




%% KWS Part 4 – Robustness sweep: accuracy vs SNR for a saved model
% Section Summary:
% Goal is to measure how test accuracy degrades as we add noise at different SNRs.
% We test 2 noise types:
%   1. White noise (randn)
%   2. Real background noises from Google’s _background_noise_ folder

% Note: We evaluate a *saved* model (no retraining for this)

% clear; clc;
% ------ 1) Picking which trained model to evaluate ----------
modelFile = 'model_MFCCEnergyZCR.mat';      % choose one of the saved model .mat files to test on
Smodel = load(modelFile); 
classes = Smodel.classes;     % class names ( {'yes','no','stop','go', 'silence'})

% ------2) Collecting test set waveforms ----------
% We load the raw preprocessed test clips (without features) so we can
% add noise to the waveform, then re-extract features inside classify_clip.
root = fullfile(pwd,'Preprocessed','test');    % path to test set clips
fileList = {};        % list of .mat files to evaluate
y_true = [];           % numeric labels for each file

for ci = 1:numel(classes)      % loop over classes
    d = dir(fullfile(root, classes{ci}, '*.mat'));
    for k = 1:numel(d)
        fileList{end+1,1} = fullfile(d(k).folder, d(k).name); 
        y_true(end+1,1) = ci; 
    end
end

% ----- 3) Defining the SNR points to test -------
% Inf dB means "no added noise" (clean baseline).
snrList = [Inf 20 10 5 0 -5];

% -----4) Loading background noise recordings -------
% This looks for Google’s "_background_noise_" folder next to Preprocessed/
% (in the directory)
bgdir = fullfile(fileparts(root), '..', '_background_noise_');
bgNoises = {};
if exist(bgdir,'dir')
    dd = dir(fullfile(bgdir,'*.wav'));
    for i=1:numel(dd)
        [nb,fsb] = audioread(fullfile(dd(i).folder, dd(i).name));
        if size(nb,2)>1, nb = mean(nb,2); end
        if fsb ~= 16000, nb = resample(nb,16000,fsb); end
        bgNoises{end+1} = nb(:); 
    end
end

% Prepare arrays to store accuracy at each SNR for both noise types
acc_white = zeros(size(snrList));
acc_bg    = zeros(size(snrList));

% --- Sweeping SNR for white noise (to represent general noisyness in the signal) ---
for si = 1:numel(snrList)
    snrdb = snrList(si);             % selects current SNR value (ex. Inf, 20, 10...)
    y_pred = zeros(size(y_true));        % vector to store predicted class IDs for all test clips


    % Loading one test waveform (.mat file created earlier)
    for n = 1:numel(fileList) 
        S = load(fileList{n}); x = S.x; Fs = S.targetFs;
        x = x(:);

        % Adding white Gaussian noise at the current SNR level
        if ~isinf(snrdb)
            x_noisy = add_noise_to_snr(x, randn(size(x)), snrdb);
        else
            x_noisy = x;     % “Inf dB” = clean (no noise)
        end

        % Classifying the noisy clip using the trained model
        out = classify_clip(modelFile, x_noisy, Fs);
        y_pred(n) = out.label_id;            % storing predicted label index
    end
    % Computign accuracy for current SNR level
    acc_white(si) = mean(y_pred == y_true);
    fprintf('White SNR %4s dB  ->  Acc = %.2f%%\n', num2str(snrdb), 100*acc_white(si));
end

% --- Sweeping SNR for background noise -------
% (for real background recordings like dishes,running tap, etc.)---
if ~isempty(bgNoises)
    for si = 1:numel(snrList)
        snrdb = snrList(si);
        y_pred = zeros(size(y_true));
        for n = 1:numel(fileList)
            S = load(fileList{n}); 
            x = S.x; Fs = S.targetFs;
            bg = bgNoises{ randi(numel(bgNoises)) };    % Randomly picks one of the background noises
            bg = bg(1:min(numel(bg),numel(x)));

            % Matching its length to the test clip
            if numel(bg) < numel(x), 
                bg = repmat(bg, ceil(numel(x)/numel(bg)), 1); 
                bg = bg(1:numel(x)); 
            end

            % Adding noise if not clean SNR (Inf)
            if ~isinf(snrdb)
                x_noisy = add_noise_to_snr(x, bg(1:numel(x)), snrdb);
            else
                x_noisy = x;
            end
            % Classifyign the noisy clip
            out = classify_clip(modelFile, x_noisy, Fs);
            y_pred(n) = out.label_id;
        end
        % Computing accuracy for current SNR level
        acc_bg(si) = mean(y_pred == y_true);
        fprintf('BG  SNR %4s dB  ->  Acc = %.2f%%\n', num2str(snrdb), 100*acc_bg(si));
    end
end

% --- Plotting accuracy vs SNR ---
figure; hold on; grid on;
plot(snrList, 100*acc_white, 'o-','LineWidth',1.8);
if any(acc_bg), plot(snrList, 100*acc_bg, 's-','LineWidth',1.8); end
xlabel('SNR (dB)'); ylabel('Accuracy (%)');
title(sprintf('Noise Robustness — %s', strrep(modelFile,'_','\_')));
legend({'White noise','Background noise'}, 'Location','southwest');
ylim([0 100]);




% -------------------------- Relevant FUnctions -----------------------------

%new
function y = add_noise_to_snr(x, noise, snrdb)

% Scale 'noise' so that SNR(x, noise) = snrdb (power SNR), then add to x.
x = x(:); noise = noise(:);
if numel(noise) < numel(x)
    noise = repmat(noise, ceil(numel(x)/numel(noise)), 1);
end
noise = noise(1:numel(x));
Px = mean(x.^2) + eps;
Pn_desired = Px / (10^(snrdb/10));
noise = noise / sqrt(mean(noise.^2) + eps) * sqrt(Pn_desired);
y = x + noise;
end


function out = classify_clip(modelFile, in, Fs_in, doPeakNorm)
% CLASSIFY_CLIP  Predict label for a .wav path or raw vector using a saved model_* .mat
% Usage:
%   out = classify_clip('model_MFCCEnergyZCR.mat','yes/abcd.wav');
%   out = classify_clip('model_MFCCEnergyZCR.mat', x, Fs);
%   out = classify_clip('model_MFCCEnergyZCR.mat', x_noisy, Fs, false); % skip normalization for robustness tests

if nargin < 4
    doPeakNorm = true;  % default: normalize to full scale
end

S = load(modelFile);                     % loads mdl_final, mu, sg, ix, classes
mdl = S.mdl_final; mu = S.mu; sg = S.sg; ix = S.ix; classes = S.classes;

% Load/prepare waveform
if ischar(in) || isstring(in)
    [x, Fs] = audioread(in);
else
    x = in; Fs = Fs_in;
end
x = x(:);
if size(x,2)>1, x = mean(x,2); end

% Match training Fs/length assumptions (16 kHz, 1 s)
targetFs = 16000; Ntarget = targetFs * 1.0;
if Fs ~= targetFs, x = resample(x, targetFs, Fs); end

% Peak normalization (skip for noisy data)
if doPeakNorm
    x = x ./ max(abs(x) + eps);
end

% --- frame params used in extractor ---
N = round(0.020*targetFs); H = round(0.010*targetFs); w = hamming(N);

% Compute features (same as training)
[E,Z] = feat_energy_zcr(x, N, H, w);
MF    = feat_mfcc_frames(x, targetFs, N, H, w, 512, 26, 13, 1e-10);
feat  = [ mean(E) std(E)  mean(Z) std(Z)  mean(MF,2).' std(MF,0,2).' ];

% Standardize + subset feature view used by this model
xs = (feat(:,ix) - mu)./max(sg,1e-12);

% Predict
[label_id, score] = predict(mdl, xs);
out.label_id = label_id;
out.label    = string(classes{label_id});
out.score    = score;
out.features = feat;
end