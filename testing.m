%% Improved_DnCNN_LSTM_Training.m
% 改進版：讓DnCNN+LSTM接近Pure ISI效果
clear; close all; clc;
testinga = 1;
%% ====================== 0) 參數設定 ======================
symbolRate = 1e3; 
Ts = 1/symbolRate;
num_samples   = 2e4;        % 訓練樣本數
signal_length = 500;        % 每條序列樣本長度
SNR_range     = [-5, 0];
sps           = 8;

% 多徑參數
channel_params.delays_sec    = [0, 2*Ts, 6*Ts];
channel_params.gains_dB      = [0, -3, -6];
channel_params.chan_len_list = [1, 1, 1];
channel_params.Ts            = Ts;
channel_params.decay         = 0.8;
channel_params.normalize_energy = true;

%% ====================== 1) 產生資料 ======================
fprintf('Generating training data...\n');
[X_train, Y_train, Sym_train, h_train] = generateTrainingDataWithDiversity( ...
    num_samples, signal_length, SNR_range, sps, channel_params);

num_val_samples = round(0.2*num_samples);
[X_val, Y_val, Sym_val, h_valid] = generateTrainingDataWithDiversity( ...
    num_val_samples, signal_length, SNR_range, sps, channel_params);

fprintf('Data generation complete.\n');

% -------- Robust normalization --------
X_train_flat = X_train(:);
p5  = prctile(X_train_flat, 5);
p95 = prctile(X_train_flat, 95);
rob = X_train_flat(X_train_flat >= p5 & X_train_flat <= p95);
train_mean = mean(rob);
train_std  = max(std(rob), 1e-6);

X_train_norm = (X_train - train_mean)/train_std;
X_val_norm   = (X_val   - train_mean)/train_std;
Y_train_norm = (Y_train - train_mean)/train_std;
Y_val_norm   = (Y_val   - train_mean)/train_std;
fprintf('Normalization complete. mean=%.4f, std=%.4f\n', train_mean, train_std);

%% ====================== 2) 改進的DnCNN架構 ======================
% 關鍵改進：使用適當的深度和濾波器數量
network_depth = 8;    % 改為8層（原本2層太淺）
num_filters   = 64;   % 減少到64（更合理的配置）

% 建立改進的DnCNN
layersDn = buildImprovedDnCNN(signal_length, network_depth, num_filters);

% 準備訓練資料
X_train_dl = X_train_norm;
Y_train_dl = X_train_norm - Y_train_norm;    % 殘差（噪聲）
X_val_dl   = X_val_norm;
Y_val_dl   = X_val_norm - Y_val_norm;

% 改進的訓練選項
optDn = trainingOptions('adam', ...
    'MaxEpochs', 30, ...              % 適中的epochs
    'MiniBatchSize', 64, ...          
    'InitialLearnRate', 5e-4, ...     % 適當的學習率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationData', {X_val_dl, Y_val_dl}, ...
    'ValidationFrequency', 100, ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 5e-4, ...      % 增加正則化
    'GradientThreshold', 1, ...
    'Verbose', true, ...
    'ExecutionEnvironment','gpu');

fprintf('Starting DnCNN training with improved architecture...\n');
netDn = trainnet(X_train_dl, Y_train_dl, layersDn, "mse", optDn);
fprintf('DnCNN training completed!\n');

%% ====================== 3) Fine-tuning DnCNN（關鍵步驟）======================
% 使用較小學習率進行fine-tuning，專注於保留ISI特徵
fprintf('Fine-tuning DnCNN for ISI preservation...\n');

% 產生專門的fine-tuning資料（較高SNR）
[X_ft, Y_ft, ~, ~] = generateTrainingDataWithDiversity( ...
    5000, signal_length, [5, 15], sps, channel_params);  % 高SNR資料

X_ft_norm = (X_ft - train_mean)/train_std;
Y_ft_norm = (Y_ft - train_mean)/train_std;

% 加權損失：更重視信號保真度
X_ft_dl = X_ft_norm;
Y_ft_dl = X_ft_norm - Y_ft_norm;

optFineTune = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-5, ...     % 很小的學習率
    'Verbose', false, ...
    'ExecutionEnvironment','gpu');

netDn = trainnet(X_ft_dl, Y_ft_dl, netDn, "mse", optFineTune);
fprintf('Fine-tuning completed!\n');

%% ====================== 9) 信號失真分析 ======================
[X_single, Y_single] = generateTrainingDataWithDiversity(1, signal_length, [5,5], sps, channel_params);
X_single_norm = (X_single - train_mean)/train_std;
noise_pred_single_norm = minibatchpredict(netDn, X_single_norm);
Y_pred_single_norm = X_single_norm - noise_pred_single_norm;
Y_pred_single_norm = postProcessDnCNN(Y_pred_single_norm, train_std);
Y_pred_single = Y_pred_single_norm * train_std + train_mean;
analyzeSignalDistortion(Y_single, X_single, Y_pred_single, signal_length);

%% ====================== 4) 建立LSTM訓練資料 ======================
K_lstm = 3; 
T_w = 2*K_lstm+1;

% --- LSTM#1（純ISI）---
[Xseq_pure_tr,  Yidx_pure_tr]  = buildSeqFromPureISI(Y_train_norm, Sym_train, sps, K_lstm);
[Xseq_pure_val, Yidx_pure_val] = buildSeqFromPureISI(Y_val_norm, Sym_val, sps, K_lstm);

% --- LSTM#2（Noisy）---
[Xseq_noisy_tr,  Yidx_noisy_tr]  = buildSeqFromPureISI(X_train_norm, Sym_train, sps, K_lstm);
[Xseq_noisy_val, Yidx_noisy_val] = buildSeqFromPureISI(X_val_norm, Sym_val, sps, K_lstm);

% --- DnCNN -> LSTM（改進：使用後處理）---
BATCH = 256;
Ntr = size(X_train_norm,4);
Xseq_dn_tr = {}; Yidx_dn_tr = [];

for st = 1:BATCH:Ntr
    ed = min(st+BATCH-1, Ntr);
    noise_pred = minibatchpredict(netDn, X_train_norm(:,:,:,st:ed));
    Ydn = X_train_norm(:,:,:,st:ed) - noise_pred;
    
    % 關鍵改進：後處理以保留ISI特徵
    Ydn = postProcessDnCNN(Ydn, train_std);
    
    [Xi, Yi] = buildSeqFromPureISI(Ydn, Sym_train(st:ed), sps, K_lstm);
    if ~isempty(Xi)
        Xseq_dn_tr = [Xseq_dn_tr, Xi]; %#ok<AGROW>
        Yidx_dn_tr = [Yidx_dn_tr; Yi(:)]; %#ok<AGROW>
    end
end

Nva = size(X_val_norm,4);
Xseq_dn_val = {}; Yidx_dn_val = [];
for st = 1:BATCH:Nva
    ed = min(st+BATCH-1, Nva);
    noise_pred = minibatchpredict(netDn, X_val_norm(:,:,:,st:ed));
    Ydn = X_val_norm(:,:,:,st:ed) - noise_pred;
    
    % 後處理
    Ydn = postProcessDnCNN(Ydn, train_std);
    
    [Xi, Yi] = buildSeqFromPureISI(Ydn, Sym_val(st:ed), sps, K_lstm);
    if ~isempty(Xi)
        Xseq_dn_val = [Xseq_dn_val, Xi]; %#ok<AGROW>
        Yidx_dn_val = [Yidx_dn_val; Yi(:)]; %#ok<AGROW>
    end
end

fprintf('Seq count -> PURE:%d / NOISY:%d / DnCNN:%d (train)\n', ...
    numel(Xseq_pure_tr), numel(Xseq_noisy_tr), numel(Xseq_dn_tr));

% 轉categorical
Ycat_pure_tr  = categorical(double(Yidx_pure_tr)+1,  1:4);
Ycat_pure_val = categorical(double(Yidx_pure_val)+1, 1:4);
Ycat_noisy_tr = categorical(double(Yidx_noisy_tr)+1, 1:4);
Ycat_noisy_val= categorical(double(Yidx_noisy_val)+1,1:4);
Ycat_dn_tr    = categorical(double(Yidx_dn_tr)+1,    1:4);
Ycat_dn_val   = categorical(double(Yidx_dn_val)+1,   1:4);

%% ====================== 5) 改進的LSTM架構 ======================
% LSTM for Pure ISI（標準版）
H1_pure = 128;
layersLSTM_pure = buildLSTMClassifier(H1_pure, 0.2);

% LSTM for Noisy（標準版）
H1_noisy = 128;
layersLSTM_noisy = buildLSTMClassifier(H1_noisy, 0.2);

% LSTM for DnCNN output
H1_dn = 128;
layersLSTM_dn = buildLSTMClassifier(H1_dn, 0.2);

% % LSTM for DnCNN output（特製版 - 更深更強）
% H1_dn = 256;  % 更多隱藏單元
% layersLSTM_dn = [
%     sequenceInputLayer(2, Name="in")
%     lstmLayer(H1_dn, OutputMode="sequence", ...
%         InputWeightsInitializer="glorot", ...
%         RecurrentWeightsInitializer="orthogonal", ...
%         BiasInitializer="unit-forget-gate", Name="lstm1")
%     layerNormalizationLayer(Name="ln1")
%     dropoutLayer(0.3, Name="drop1")  % 更多dropout
%     lstmLayer(H1_dn/2, OutputMode="sequence", ...
%         InputWeightsInitializer="glorot", ...
%         RecurrentWeightsInitializer="orthogonal", ...
%         BiasInitializer="unit-forget-gate", Name="lstm2")
%     layerNormalizationLayer(Name="ln2")
%     dropoutLayer(0.3, Name="drop2")
%     lstmLayer(H1_dn/4, OutputMode="last", ...
%         InputWeightsInitializer="glorot", ...
%         RecurrentWeightsInitializer="orthogonal", ...
%         BiasInitializer="unit-forget-gate", Name="lstm3")
%     layerNormalizationLayer(Name="ln3")
%     dropoutLayer(0.2, Name="drop3")
%     fullyConnectedLayer(4, Name="fc")
%     softmaxLayer(Name="sm")
% ];
% 
% % 訓練選項
optsL_base = trainingOptions("adam", ...
    MaxEpochs=15, MiniBatchSize=256, ...
    InitialLearnRate=1e-3, ...
    SequenceLength="longest", ...
    Shuffle="every-epoch", ...
    Metrics="accuracy", ...
    GradientThreshold=1, ...
    L2Regularization=1e-3, ...
    ExecutionEnvironment="gpu", ...
    Verbose=true);
% 
% % DnCNN專用LSTM訓練選項（更多epochs）
% optsL_dn = trainingOptions("adam", ...
%     MaxEpochs=25, MiniBatchSize=128, ...  % 更多epochs，更小batch
%     InitialLearnRate=5e-4, ...             % 較小學習率
%     LearnRateSchedule='piecewise', ...
%     LearnRateDropPeriod=8, ...
%     LearnRateDropFactor=0.5, ...
%     SequenceLength="longest", ...
%     Shuffle="every-epoch", ...
%     Metrics="accuracy", ...
%     GradientThreshold=1, ...
%     L2Regularization=5e-4, ...
%     ExecutionEnvironment="gpu", ...
%     Verbose=true);

%% ====================== 6) 訓練LSTM ======================
fprintf('Training LSTM#1 (PURE-ISI upper bound)...\n');
netLSTM_pure = trainnet(Xseq_pure_tr, Ycat_pure_tr, layersLSTM_pure, "crossentropy", ...
    setVal(optsL_base, Xseq_pure_val, Ycat_pure_val));

fprintf('Training LSTM#2 (NOISY baseline)...\n');
netLSTM_noisy = trainnet(Xseq_noisy_tr, Ycat_noisy_tr, layersLSTM_noisy, "crossentropy", ...
    setVal(optsL_base, Xseq_noisy_val, Ycat_noisy_val));

fprintf('Training specialized LSTM for DnCNN outputs...\n');
netLSTM_dn = trainnet(Xseq_dn_tr, Ycat_dn_tr, layersLSTM_dn, "crossentropy", ...
    setVal(optsL_base, Xseq_dn_val, Ycat_dn_val));
% netLSTM_dn = trainnet(Xseq_dn_tr, Ycat_dn_tr, layersLSTM_dn, "crossentropy", ...
%     setVal(optsL_dn, Xseq_dn_val, Ycat_dn_val));

%% ====================== 7) 評估：SER vs SNR ======================
test_SNRs = -5:1:15;
num_test  = 1000;
SER_pure  = nan(numel(test_SNRs),1);
SER_noisy = nan(numel(test_SNRs),1);
SER_dn    = nan(numel(test_SNRs),1);

for si = 1:numel(test_SNRs)
    snr = test_SNRs(si);
    [Xte, Yte, Sym_te] = generateTrainingDataWithDiversity( ...
        num_test, signal_length, [snr snr], sps, channel_params);
    
    Xte_n = (Xte - train_mean)/train_std;
    Yte_n = (Yte - train_mean)/train_std;
    
    % Build sequences
    [S_pure,  L_pure ] = buildSeqFromPureISI(Yte_n, Sym_te, sps, K_lstm);
    [S_noisy, L_noisy] = buildSeqFromPureISI(Xte_n, Sym_te, sps, K_lstm);
    
    % DnCNN denoise with post-processing
    noise_pred = minibatchpredict(netDn, Xte_n);
    Ydn_n = Xte_n - noise_pred;
    Ydn_n = postProcessDnCNN(Ydn_n, train_std);  % 後處理
    [S_dn, L_dn] = buildSeqFromPureISI(Ydn_n, Sym_te, sps, K_lstm);
    
    % 預測與SER
    SER_pure(si)  = 1 - acc_from_probs(minibatchpredict(netLSTM_pure,  S_pure),  L_pure );
    SER_noisy(si) = 1 - acc_from_probs(minibatchpredict(netLSTM_noisy, S_noisy), L_noisy);
    SER_dn(si)    = 1 - acc_from_probs(minibatchpredict(netLSTM_dn,    S_dn),    L_dn   );
    
    fprintf('SNR=%2d dB | SER: PURE=%.3g NOISY=%.3g DnCNN=%.3g | Gap=%.3g\n', ...
            snr, SER_pure(si), SER_noisy(si), SER_dn(si), SER_dn(si)-SER_pure(si));
end

%% ====================== 8) 視覺化結果 ======================
figure('Position',[70 70 1200 600]);

subplot(1,2,1);
semilogy(test_SNRs, SER_pure, '-^', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0 0.5 0]);
hold on;
semilogy(test_SNRs, SER_noisy,'-o', 'LineWidth', 2.0, 'MarkerSize', 9, 'Color', [1 0 0]);
semilogy(test_SNRs, SER_dn,   '-s', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0 0 1]);
grid on; xlabel('SNR (dB)'); ylabel('SER');
title('SER vs SNR Performance Comparison');
legend('LSTM#1: PURE-ISI (Upper Bound)', ...
       'LSTM#2: ISI+AWGN (Baseline)', ...
       'Improved DnCNN→LSTM', ...
       'Location','southwest');
ylim([1e-5, 1]);

subplot(1,2,2);
% 顯示改善程度
improvement_over_noisy = (SER_noisy - SER_dn) ./ SER_noisy * 100;
gap_to_pure = (SER_dn - SER_pure) ./ SER_pure * 100;

yyaxis left;
plot(test_SNRs, improvement_over_noisy, '-o', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('Improvement over Noisy (%)');
ylim([0 100]);

yyaxis right;
plot(test_SNRs, gap_to_pure, '-s', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('Gap to Pure ISI (%)');
ylim([0 200]);

xlabel('SNR (dB)');
title('Performance Analysis');
legend('Improvement over baseline', 'Gap to upper bound', 'Location', 'best');
grid on;

%% ====================== 輔助函式 ======================

function layers = buildImprovedDnCNN(signal_length, depth, num_filters)
    % 改進的DnCNN架構，使用殘差連接
    layers = [
        imageInputLayer([signal_length 1 2], 'Name','input', 'Normalization','none')
        convolution2dLayer([7 1], num_filters, 'Padding','same', ...
            'Name','conv1', 'WeightsInitializer','he')  % 較大的核
        reluLayer('Name','relu1')
    ];
    
    % 中間層使用不同大小的卷積核
    for i = 2:(depth-1)
        if mod(i,2) == 0
            kernel_size = [5 1];
        else
            kernel_size = [3 1];
        end
        
        layers = [layers
            convolution2dLayer(kernel_size, num_filters, 'Padding','same', ...
                'Name',['conv' num2str(i)], 'WeightsInitializer','he')
            batchNormalizationLayer('Name',['bn' num2str(i)])
            reluLayer('Name',['relu' num2str(i)])
        ]; %#ok<AGROW>
    end
    
    % 輸出層
    layers = [layers
        convolution2dLayer([5 1], 2, 'Padding','same', ...
            'Name',['conv' num2str(depth)], 'WeightsInitializer','he')
    ];
end

function Ydn_processed = postProcessDnCNN(Ydn, train_std)
    % 後處理：輕微平滑以保留ISI特徵
    [H, W, C, N] = size(Ydn);
    Ydn_processed = Ydn;
    
    % 設計簡單的平滑濾波器
    smooth_kernel = [0.1, 0.8, 0.1]';  % 輕微平滑
    
    for n = 1:N
        for c = 1:C
            signal = squeeze(Ydn(:,1,c,n));
            % 輕微平滑
            signal_smooth = conv(signal, smooth_kernel, 'same');
            % 混合原始和平滑信號
            alpha = 0.7;  % 70%原始，30%平滑
            Ydn_processed(:,1,c,n) = alpha * signal + (1-alpha) * signal_smooth;
        end
    end
end

function layers = buildLSTMClassifier(hidden_units, dropout_rate)
    layers = [
        sequenceInputLayer(2, Name="in")
        lstmLayer(hidden_units, OutputMode="sequence", ...
            InputWeightsInitializer="glorot", ...
            RecurrentWeightsInitializer="orthogonal", ...
            BiasInitializer="unit-forget-gate", Name="lstm1")
        layerNormalizationLayer(Name="ln1")
        dropoutLayer(dropout_rate, Name="drop1")
        lstmLayer(hidden_units, OutputMode="last", ...
            InputWeightsInitializer="glorot", ...
            RecurrentWeightsInitializer="orthogonal", ...
            BiasInitializer="unit-forget-gate", Name="lstm2")
        layerNormalizationLayer(Name="ln2")
        dropoutLayer(dropout_rate, Name="drop2")
        fullyConnectedLayer(4, Name="fc")
        softmaxLayer(Name="sm")
    ];
end

function opts2 = setVal(opts, Xval, Yval)
    opts2 = opts;
    opts2.ValidationData = {Xval, Yval};
    opts2.ValidationFrequency = 50;
end

function acc = acc_from_probs(probs, Yidx0)
    if isempty(probs) || isempty(Yidx0)
        acc = NaN; return;
    end
    if size(probs,2)==4
        [~, pred] = max(probs,[],2);
    else
        [~, pred] = max(probs,[],1); pred = pred(:);
    end
    pred = double(pred)-1;
    acc = mean(pred == double(Yidx0));
end

% [保留原始的generateTrainingDataWithDiversity函式]
function [X, Y, Sym, h_int] = generateTrainingDataWithDiversity(num_samples, signal_length, SNR_range, sps, channel_params)
    X   = zeros(signal_length, 1, 2, num_samples, 'single');
    Y   = zeros(signal_length, 1, 2, num_samples, 'single');
    Sym = cell(num_samples,1);

    K = floor(signal_length/sps);
    bits_per_symbol = 2;

    if nargin < 5 || isempty(channel_params)
        delays_sym      = [0 1 2];
        gains_dB        = [0 -3 -6];
        chan_len_list   = [1 1 1];
        decay           = 0.85;
        normalize_energy = true;
        Ts = 1;
    else
        if isfield(channel_params,'delays_sym')
            delays_sym = channel_params.delays_sym;
        elseif isfield(channel_params,'delays_sec') && isfield(channel_params,'Ts')
            delays_sym = channel_params.delays_sec ./ channel_params.Ts;
        else
            error('channel_params needs delays_sym or delays_sec+Ts.');
        end
        gains_dB = channel_params.gains_dB;
        if isfield(channel_params,'chan_len_list'), chan_len_list = channel_params.chan_len_list; else, chan_len_list = ones(size(gains_dB)); end
        if isfield(channel_params,'decay'), decay = channel_params.decay; else, decay = 0.85; end
        if isfield(channel_params,'normalize_energy'), normalize_energy = channel_params.normalize_energy; else, normalize_energy = true; end
    end

    delay_samp = round(delays_sym * sps);
    gains_lin  = 10.^(gains_dB/20);

    Lh = max(delay_samp) + max(chan_len_list);
    h_int = zeros(Lh,1);
    for p = 1:numel(delay_samp)
        start_idx = delay_samp(p) + 1;
        M = chan_len_list(p);
        taps = decay.^(0:M-1);
        taps = taps / sqrt(sum(abs(taps).^2)+eps);
        h_int(start_idx:start_idx+M-1) = h_int(start_idx:start_idx+M-1) + gains_lin(p)*taps(:);
    end
    if normalize_energy
        h_int = h_int / sqrt(sum(abs(h_int).^2)+eps);
    end

    for i = 1:num_samples
        data_bits = randi([0,1], K*bits_per_symbol, 1);
        qpsk_symbols = zeros(K,1); 
        sym_id = zeros(K,1);
        for k = 1:K
            b1 = data_bits(2*k-1); b2 = data_bits(2*k);
            if     b1==0 && b2==0, qpsk_symbols(k) = -1 - 1j; sym_id(k)=1;
            elseif b1==0 && b2==1, qpsk_symbols(k) = -1 + 1j; sym_id(k)=2;
            elseif b1==1 && b2==0, qpsk_symbols(k) =  1 - 1j; sym_id(k)=3;
            else,                   qpsk_symbols(k) =  1 + 1j; sym_id(k)=4;
            end
        end
        qpsk_symbols = qpsk_symbols / sqrt(2);
        Sym{i} = sym_id;

        upsampled = upsample(qpsk_symbols, sps);
        beta = 0.25; spanInSymbols = 10;
        pulse = rcosdesign(beta, spanInSymbols, sps, "sqrt");
        bb = filter(pulse, 1, upsampled);

        isi = filter(h_int, 1, bb);
        if numel(isi) >= signal_length
            isi = isi(1:signal_length);
        else
            isi = [isi; zeros(signal_length - numel(isi),1)];
        end

        if rand()<0.3
            if rand()<0.5, snr = SNR_range(1) + 5*rand();
            else,           snr = SNR_range(2) - 5*rand();
            end
        else
            snr = SNR_range(1) + (SNR_range(2)-SNR_range(1))*rand();
        end
        noisy = awgn(isi, snr, 'measured');

        Y(:,1,1,i) = single(real(isi));
        Y(:,1,2,i) = single(imag(isi));
        X(:,1,1,i) = single(real(noisy));
        X(:,1,2,i) = single(imag(noisy));
    end
end

% [保留原始的buildSeqFromPureISI和analyzeSignalDistortion函式]
function [Xseq, Yidx0] = buildSeqFromPureISI(Y_norm, Sym_cell, sps, K_lstm)
    N   = size(Y_norm,4);
    T_w = 2*K_lstm + 1;
    W = zeros(N,1);
    for i = 1:N
        Ki = numel(Sym_cell{i});
        W(i) = max(0, Ki - 2*K_lstm);
    end
    Mtot = sum(W);
    Xseq = cell(1, Mtot);
    Yidx0 = zeros(Mtot,1);
    wptr = 1;

    for i = 1:N
        if W(i)==0, continue; end
        I = Y_norm(:,1,1,i); Q = Y_norm(:,1,2,i);
        Ki = numel(Sym_cell{i});
        M_eff = Ki - 2*K_lstm; 
        if M_eff <= 0, continue; end
        for c = 1:M_eff
            center_sym = c + K_lstm;
            start_sample = (c-1) * sps + 1;
            end_sample = start_sample + T_w * sps - 1;
            if end_sample > length(I), break; end
            Xi = [double(I(start_sample:end_sample)), double(Q(start_sample:end_sample))];
            Xseq{wptr} = single(Xi);
            Yidx0(wptr) = double(Sym_cell{i}(center_sym)) - 1;
            wptr = wptr + 1;
        end
    end
    if wptr <= Mtot
        Xseq  = Xseq(1:wptr-1);
        Yidx0 = Yidx0(1:wptr-1);
    end
end

function analyzeSignalDistortion(Y_pure, X_noisy, Y_denoised, signal_length)
    figure('Position', [100, 100, 1400, 700]);
    
    subplot(2,3,1);
    t = 1:min(200, signal_length);
    plot(t, squeeze(Y_pure(t,1,1,1)), 'b-', 'LineWidth', 1.5); hold on;
    plot(t, squeeze(Y_denoised(t,1,1,1)), 'g--', 'LineWidth', 1.5);
    xlabel('Sample Index'); ylabel('Amplitude');
    title('Pure ISI vs Denoised Signal');
    legend('Pure ISI', 'Denoised', 'Location', 'best');
    grid on;
    
    subplot(2,3,2);
    error_noisy = abs(squeeze(X_noisy(:,1,1,1)) - squeeze(Y_pure(:,1,1,1)));
    error_denoised = abs(squeeze(Y_denoised(:,1,1,1)) - squeeze(Y_pure(:,1,1,1)));
    plot(t, error_noisy(t), 'r-', 'LineWidth', 0.8, 'Color', [1,0,0,0.5]); hold on;
    plot(t, error_denoised(t), 'g-', 'LineWidth', 1.5);
    xlabel('Sample Index'); ylabel('Absolute Error');
    title('Denoising Error Reduction');
    legend('Error (Noisy)','Error (Denoised)','Location','best');
    grid on;
    
    subplot(2,3,3);
    error_denoised_all = [abs(squeeze(Y_denoised(:,1,1,1)) - squeeze(Y_pure(:,1,1,1))); ...
                          abs(squeeze(Y_denoised(:,1,2,1)) - squeeze(Y_pure(:,1,2,1)))];
    error_noisy_all = [abs(squeeze(X_noisy(:,1,1,1)) - squeeze(Y_pure(:,1,1,1))); ...
                       abs(squeeze(X_noisy(:,1,2,1)) - squeeze(Y_pure(:,1,2,1)))];
    histogram(error_denoised_all, 50, 'FaceColor', 'g', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    hold on;
    histogram(error_noisy_all, 50, 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    xlabel('Error Magnitude'); ylabel('Count');
    title('Error Distribution');
    legend('Denoised', 'Noisy');
    grid on;
    
    subplot(2,3,4);
    f = linspace(-0.5, 0.5, signal_length);
    Y_pure_fft = abs(fftshift(fft(squeeze(Y_pure(:,1,1,1)))));
    Y_denoised_fft = abs(fftshift(fft(squeeze(Y_denoised(:,1,1,1)))));
    plot(f, 20*log10(Y_pure_fft+eps), 'b-', 'LineWidth', 1.5); hold on;
    plot(f, 20*log10(Y_denoised_fft+eps), 'g--', 'LineWidth', 1.5);
    xlabel('Normalized Frequency'); ylabel('Magnitude (dB)');
    title('Frequency Response Preservation');
    legend('Pure ISI','Denoised');
    grid on;
    
    subplot(2,3,5);
    Y_pure_phase = angle(fftshift(fft(squeeze(Y_pure(:,1,1,1)))));
    Y_denoised_phase = angle(fftshift(fft(squeeze(Y_denoised(:,1,1,1)))));
    plot(f, Y_pure_phase, 'b-', 'LineWidth', 1.5); hold on;
    plot(f, Y_denoised_phase, 'g--', 'LineWidth', 1.5);
    xlabel('Normalized Frequency'); ylabel('Phase (rad)');
    title('Phase Response Preservation');
    legend('Pure ISI','Denoised');
    grid on;
    
    subplot(2,3,6);
    [xcorr_vals, lags] = xcorr(squeeze(Y_pure(:,1,1,1)), squeeze(Y_denoised(:,1,1,1)), 50, 'normalized');
    plot(lags, xcorr_vals, 'k-', 'LineWidth', 2);
    xlabel('Lag'); ylabel('Correlation');
    title(sprintf('Cross-Correlation (Peak = %.4f)', max(xcorr_vals)));
    grid on;
    
    sgtitle('Comprehensive Signal Distortion Analysis', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 計算統計
    Y_pure_complex = squeeze(Y_pure(:,1,1,1)) + 1j*squeeze(Y_pure(:,1,2,1));
    Y_denoised_complex = squeeze(Y_denoised(:,1,1,1)) + 1j*squeeze(Y_denoised(:,1,2,1));
    X_noisy_complex = squeeze(X_noisy(:,1,1,1)) + 1j*squeeze(X_noisy(:,1,2,1));
    
    corr_coef = corrcoef(abs(Y_pure_complex), abs(Y_denoised_complex));
    noise_before = X_noisy_complex - Y_pure_complex;
    noise_after = Y_denoised_complex - Y_pure_complex;
    snr_improvement = 10*log10(mean(abs(noise_before).^2) / (mean(abs(noise_after).^2)+eps));
    nmse = mean(abs(Y_denoised_complex - Y_pure_complex).^2) / mean(abs(Y_pure_complex).^2);
    sdr = 10*log10(mean(abs(Y_pure_complex).^2) / (mean(abs(Y_denoised_complex - Y_pure_complex).^2)+eps));
    var_ratio = var(Y_denoised_complex) / var(Y_pure_complex);
    
    fprintf('\n=== Improved DnCNN Distortion Summary ===\n');
    fprintf('Correlation: %.4f | SNR Improvement: %.2f dB | NMSE: %.6f\n', ...
        corr_coef(1,2), snr_improvement, nmse);
    fprintf('SDR: %.2f dB | Variance Ratio: %.4f\n', sdr, var_ratio);
    fprintf('==========================================\n');
end