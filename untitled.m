% --- 腳本主程式 ---
clear; clc; close all;
disp("開始執行 1D-CNN vs 1D-DnCNN 在高SNR下的失真比較...");

% =========================================================================
% --- 1. 參數設定 (修改為 1D 訊號) ---
% =========================================================================
symbolRate = 1e3; 
Ts = 1/symbolRate;
numTrainSamples = 20000;      % 訓練樣本數
signal_length = 500;      % 信號長度 (CNN一次看的sample個數)
sps = 16;                 % 每符號採樣數
Fs = symbolRate * sps;    % 取樣率

% *** 關鍵：使用寬廣的SNR範圍進行訓練，包含低SNR ***
snr_dB_train_range = [-5, 20]; 
fprintf("訓練SNR範圍: [%d, %d] dB\n", snr_dB_train_range(1), snr_dB_train_range(2));

% 多徑通道參數 (來自您之前的腳本)
channel_params.delays_sec = [0, 2*Ts, 6*Ts];
channel_params.gains_dB = [0, -3, -5];
channel_params.chan_len_list = [1, 1, 1];
channel_params.Ts = Ts;
channel_params.decay = 0.8;
channel_params.normalize_energy = true;

% =========================================================================
% --- 2. 訓練資料生成 (修改為 1D 訊號) ---
% =========================================================================
disp("正在生成 1D QPSK 訓練資料...");
% *** 關鍵：使用 1D 訊號生成函數 ***
[XTrain, YTrain, ~, ~] = generateTrainingDataWithDiversity(numTrainSamples, signal_length, snr_dB_train_range, sps, channel_params, Fs);
XTrain = single(XTrain);
YTrain = single(YTrain);
ZTrain = XTrain - YTrain; % ZTrain = trueNoise (噪聲目標)

% 數據正規化 (與之前相同)
X_flat = XTrain(:);
p5 = prctile(X_flat, 5); p95 = prctile(X_flat, 95);
robust_data = X_flat(X_flat >= p5 & X_flat <= p95);
data_mean = mean(robust_data);
data_std = max(std(robust_data), 1e-6);

% 正規化 X (輸入)
XTrain_norm = (XTrain - data_mean) / data_std;
% 正規化 Y (標準CNN目標)
YTrain_norm = (YTrain - data_mean) / data_std;
% 正規化 Z (DnCNN目標)
ZTrain_norm = (ZTrain - data_mean) / data_std; % 注意：噪聲也需要正規化

% --- 建立 Datastore ---
dsX = arrayDatastore(XTrain_norm, 'IterationDimension', 4);
dsY = arrayDatastore(YTrain_norm, 'IterationDimension', 4);
dsZ = arrayDatastore(ZTrain_norm, 'IterationDimension', 4);

dsTrainX_Y = combine(dsX, dsY); % (noisy, clean)
dsTrainX_Z = combine(dsX, dsZ); % (noisy, noise)
disp("資料準備完成。");

% =========================================================================
% --- 3. 網路定義 (*** 修改為 1D 卷積架構 ***) ---
% =========================================================================
layers = [
    imageInputLayer([signal_length 1 2], Normalization="none", Name="input")

    % 1D 卷積模塊 1
    convolution2dLayer([7 1], 32, Padding="same", Name="conv1", WeightsInitializer="he")
    batchNormalizationLayer(Name="bn1")
    reluLayer(Name="relu1")
    maxPooling2dLayer([2 1], Stride=[1 1], Padding="same", Name="pool1")

    % 1D 卷積模塊 2
    convolution2dLayer([5 1], 64, Padding="same", Name="conv2", WeightsInitializer="he")
    batchNormalizationLayer(Name="bn2")
    reluLayer(Name="relu2")
    maxPooling2dLayer([2 1], Stride=[1 1], Padding="same", Name="pool2")

    % 1D 卷積模塊 3
    convolution2dLayer([5 1], 128, Padding="same", Name="conv3", WeightsInitializer="he")
    batchNormalizationLayer(Name="bn3")
    reluLayer(Name="relu3")

    % 1D 卷積模塊 4
    convolution2dLayer([3 1], 64, Padding="same", Name="conv4", WeightsInitializer="he")
    batchNormalizationLayer(Name="bn4")
    reluLayer(Name="relu4")
    
    % 輸出層
    convolution2dLayer([3 1], 2, Padding="same", Name="conv_out", WeightsInitializer="glorot")
];

% 建立 LayerGraph
lgraph = layerGraph(layers);
% 轉換為 dlnetwork
net = dlnetwork(lgraph);
disp("網路架構(1D-CNN)建立完成。");

% --- 4. 訓練選項設定 ---
opts = trainingOptions("adam", ...
    MiniBatchSize=128, ...
    MaxEpochs=30, ... % (1D訊號可能需要稍多一點Epochs)
    InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=10, ... % 提早降低學習率
    LearnRateDropFactor=0.5, ...
    Shuffle="every-epoch", ...
    Verbose=true, ...
    ExecutionEnvironment="gpu", ... 
    OutputNetwork="auto");

% --- 5. 訓練網路 ---
disp("開始訓練網路...");
% --- 訓練模型一：標準 CNN (學習乾淨訊號 Y) ---
disp("--- 正在訓練 標準 CNN (學習目標: Y_clean) ---");
trainedNetX_Y = trainnet(dsTrainX_Y, net, "mse", opts);
% --- 訓練模型二：DnCNN (學習噪聲 Z) ---
disp("--- 正在訓練 DnCNN (學習目標: Z_noise) ---");
trainedNetX_Z = trainnet(dsTrainX_Z, net, "mse", opts);
disp("訓練完成！");

% =========================================================================
% === 6. 訓練後效能驗證與繪圖 (*** 關鍵測試 ***) ===
% =========================================================================
disp("開始進行網路效能測試...");
% 1. 生成新的測試資料
numTestFrames = 1000;
% *** 關鍵：使用極高 SNR 進行測試 ***
snr_dB_test = 40; 
fprintf("測試 SNR: %d dB (高信噪比測試)\n", snr_dB_test);
[XTestNoisy, YTestClean] = generateTrainingDataWithDiversity(numTestFrames, signal_length, snr_dB_test, sps, channel_params, Fs);

% 2. 使用 trainedNet 進行預測 (去躁)
% 正規化測試輸入
dlXTestNoisy = dlarray((XTestNoisy - data_mean) / data_std, 'SSCB');
% 準備好用於計算 MSE 的乾淨目標 (正規化)
dlYTestClean = dlarray((YTestClean - data_mean) / data_std, 'SSCB');

% --- 執行模型一 (CNN) 預測 ---
dlYPredClean_norm = predict(trainedNetX_Y, dlXTestNoisy);
% --- 執行模型二 (DnCNN) 預測 ---
dlYPredNoise_norm = predict(trainedNetX_Z, dlXTestNoisy);

% 從 GPU 和 dlarray 中取回資料
YPredClean_norm = extractdata(dlYPredClean_norm); % CNN 的輸出 (預測的乾淨訊號)
YPredNoise_norm = extractdata(dlYPredNoise_norm); % DnCNN 的輸出 (預測的噪聲)

% 3. 處理預測結果

% --- 模型一 (CNN) 的去噪結果 ---
% (已是正規化後的乾淨訊號)
denoised_I_CNN = YPredClean_norm(:,:,1,:);
denoised_Q_CNN = YPredClean_norm(:,:,2,:);

% --- 模型二 (DnCNN) 的去噪結果 ---
% 關鍵一步：去噪訊號 = 原始噪聲訊號 - 預測的噪聲
% (所有操作都在正規化域中進行)
XTestNoisy_norm = extractdata(dlXTestNoisy);
denoised_I_DnCNN = XTestNoisy_norm(:,:,1,:) - YPredNoise_norm(:,:,1,:);
denoised_Q_DnCNN = XTestNoisy_norm(:,:,2,:) - YPredNoise_norm(:,:,2,:);

% --- 準備Ground Truth 和 Noisy (用於比較) ---
noisy_I = XTestNoisy_norm(:,:,1,:);
noisy_Q = XTestNoisy_norm(:,:,2,:);
clean_I = extractdata(dlYTestClean(:,:,1,:));
clean_Q = extractdata(dlYTestClean(:,:,2,:));

% --- 轉換為 1D 複數序列 (*** 已修正 Reshape 邏輯 ***) ---
% 將 [500, 1, 1000] 壓平為 [500000, 1]
noisy_sym = reshape(noisy_I, [], 1) + 1j * reshape(noisy_Q, [], 1);
clean_sym = reshape(clean_I, [], 1) + 1j * reshape(clean_Q, [], 1);
denoised_sym_CNN = reshape(denoised_I_CNN, [], 1) + 1j * reshape(denoised_Q_CNN, [], 1);
denoised_sym_DnCNN = reshape(denoised_I_DnCNN, [], 1) + 1j * reshape(denoised_Q_DnCNN, [], 1);

% --- 計算 MSE 進行定量比較 (在正規化域中) ---
mse_CNN = mean(abs(denoised_sym_CNN - clean_sym).^2);
mse_DnCNN = mean(abs(denoised_sym_DnCNN - clean_sym).^2);
mse_Noisy = mean(abs(noisy_sym - clean_sym).^2);

fprintf("\n--- 高 SNR (%d dB) 測試結果 (正規化域 MSE) ---\n", snr_dB_test);
fprintf("原始噪聲 MSE: %.6f\n", mse_Noisy);
fprintf("標準 CNN 輸出 MSE: %.6f  (*** 預期此值較高, 代表失真 ***)\n", mse_CNN);
fprintf("DnCNN 輸出 MSE: %.6f  (*** 預期此值接近噪聲MSE, 代表完美還原 ***)\n", mse_DnCNN);

% 4. 繪製星座圖 (Constellation Plot) - 2x2 比較圖
figure;
set(gcf, 'Position', [50, 50, 1000, 800]); 
% (為避免點太多，只畫前 20000 個符號)
numPlotPoints = min(20000, length(noisy_sym));
idxPlot = 1:numPlotPoints;

subplot(2,2,1);
plot(noisy_sym(idxPlot), '.r'); % 紅色
title(sprintf('Noisy Constellation (SNR = %d dB)', snr_dB_test));
axis equal; grid on; xlim([-2 2]); ylim([-2 2]);
xlabel('I-Channel'); ylabel('Q-Channel');

subplot(2,2,2);
plot(clean_sym(idxPlot), '.k'); % 黑色
title('Clean Constellation (Ground Truth)');
axis equal; grid on; xlim([-2 2]); ylim([-2 2]);
xlabel('I-Channel'); ylabel('Q-Channel');

subplot(2,2,3);
plot(denoised_sym_CNN(idxPlot), '.b'); % 藍色
title('Denoised (標準 CNN) - 預期有失真', 'Color', 'b');
axis equal; grid on; xlim([-2 2]); ylim([-2 2]);
xlabel('I-Channel'); ylabel('Q-Channel');

subplot(2,2,4);
plot(denoised_sym_DnCNN(idxPlot), '.g'); % 綠色
title('Denoised (DnCNN) - 預期完美還原', 'Color', 'g');
axis equal; grid on; xlim([-2 2]); ylim([-2 2]);
xlabel('I-Channel'); ylabel('Q-Channel');

sgtitle('CNN vs DnCNN 在高SNR下的失真比較 (1D 訊號處理)', 'FontSize', 16, 'FontWeight', 'bold');

% 5. 繪製波形圖 (Waveform Plot) - *** 已修正為 1D 樣本 ***
figure;
set(gcf, 'Position', [50, 550, 1200, 600]);
numSamplesToPlot = 200; % 繪製前 200 個樣本
sample_idx = 1:numSamplesToPlot;

% 提取第 1 幀訊號用於繪圖
i_frame_plot = 1;
clean_I_plot = clean_I(:,:,i_frame_plot);
noisy_I_plot = noisy_I(:,:,i_frame_plot);
denoised_I_CNN_plot = denoised_I_CNN(:,:,i_frame_plot);
denoised_I_DnCNN_plot = denoised_I_DnCNN(:,:,i_frame_plot);

clean_Q_plot = clean_Q(:,:,i_frame_plot);
noisy_Q_plot = noisy_Q(:,:,i_frame_plot);
denoised_Q_CNN_plot = denoised_Q_CNN(:,:,i_frame_plot);
denoised_Q_DnCNN_plot = denoised_Q_DnCNN(:,:,i_frame_plot);

% 繪製 I 通道
subplot(2,1,1);
plot(sample_idx, clean_I_plot(sample_idx), 'k-', 'LineWidth', 2, 'DisplayName', 'Clean (I) (Ground Truth)');
hold on;
% plot(sample_idx, noisy_I_plot(sample_idx), 'r:', 'DisplayName', 'Noisy (I)');
plot(sample_idx, denoised_I_CNN_plot(sample_idx), 'b--', 'LineWidth', 1.5, 'DisplayName', 'CNN (I) - 預期振幅衰減');
plot(sample_idx, denoised_I_DnCNN_plot(sample_idx), 'g-.', 'LineWidth', 1.5, 'DisplayName', 'DnCNN (I) - 預期完美重合');
title('I-Channel Waveform (高SNR測試)');
legend('Location', 'best'); grid on;
ylabel('Amplitude');

% 繪製 Q 通道
subplot(2,1,2);
plot(sample_idx, clean_Q_plot(sample_idx), 'k-', 'LineWidth', 2, 'DisplayName', 'Clean (Q) (Ground Truth)');
hold on;
% plot(sample_idx, noisy_Q_plot(sample_idx), 'r:', 'DisplayName', 'Noisy (Q)');
plot(sample_idx, denoised_Q_CNN_plot(sample_idx), 'b--', 'LineWidth', 1.5, 'DisplayName', 'CNN (Q) - 預期振幅衰減');
plot(sample_idx, denoised_Q_DnCNN_plot(sample_idx), 'g-.', 'LineWidth', 1.5, 'DisplayName', 'DnCNN (Q) - 預期完美重合');
title('Q-Channel Waveform (高SNR測試)');
legend('Location', 'best'); grid on;
xlabel('Sample Index');
ylabel('Amplitude');

sgtitle('CNN vs DnCNN 波形失真比較 (1D 訊號處理)', 'FontSize', 16, 'FontWeight', 'bold');

disp('驗證繪圖完成！');


% =========================================================================
% === 輔助函數 (Local Functions) ===
% =========================================================================

% *** 輔助函數 1：1D 訊號生成 (來自您的 CNN_LSTM 腳本) ***
function [X, Y, Sym, h_total] = generateTrainingDataWithDiversity(num_samples, signal_length, snr_input, sps, channel_params, Fs)
    % 輸出: X (noisy), Y (clean), Sym (labels)
    % 數據維度: [signal_length, 1, 2, num_samples]
    
    X = zeros(signal_length, 1, 2, num_samples, 'single');
    Y = zeros(signal_length, 1, 2, num_samples, 'single');
    Sym = cell(num_samples, 1);
    
    K = floor(signal_length / sps);
    bits_per_symbol = 2;

    % --- 通道參數 ---
    Ts = channel_params.Ts;
    delays_sym = channel_params.delays_sec ./ Ts;
    gains_dB = channel_params.gains_dB;
    chan_len_list = channel_params.chan_len_list;
    decay = channel_params.decay;
    normalize_energy = channel_params.normalize_energy;
    
    % --- 建立多路徑通道 ---
    delay_samp = round(delays_sym * sps);
    gains_lin = 10.^(gains_dB/20);
    Lh = max(delay_samp) + max(chan_len_list);
    h_total = zeros(Lh, 1);
    for p = 1:numel(delay_samp)
        start_idx = delay_samp(p) + 1;
        M = chan_len_list(p);
        taps = decay.^(0:M-1);
        taps = taps / sqrt(sum(abs(taps).^2) + eps);
        h_total(start_idx:start_idx+M-1) = h_total(start_idx:start_idx+M-1) + gains_lin(p)*taps(:);
    end
    if normalize_energy
        h_total = h_total / sqrt(sum(abs(h_total).^2) + eps);
    end

    % --- RRC filter (對稱群延遲) ---
    beta = 0.5;
    spanInSymbols = 10;
    pulse = rcosdesign(beta, spanInSymbols, sps, "sqrt");
    rrc_delay = (length(pulse)-1)/2;  % 群延遲 (samples)
    chan_main_delay = find(abs(h_total) == max(abs(h_total)), 1) - 1;  % 主徑樣本延遲
    total_delay_samples = round(rrc_delay + chan_main_delay);

    % --- 判斷 SNR 輸入是範圍還是純量 ---
    if isscalar(snr_input)
        snr_is_range = false;
        snr_dB_fixed = snr_input;
    else
        snr_is_range = true;
        snr_range = snr_input;
    end
    
    % --- 產生訓練樣本 ---
    for i = 1:num_samples
        % QPSK bits & symbols
        data_bits = randi([0,1], K*bits_per_symbol, 1);
        qpsk_symbols = zeros(K,1);
        sym_id = zeros(K,1);
        for k = 1:K
            b1 = data_bits(2*k-1); b2 = data_bits(2*k);
            if b1==0 && b2==0, qpsk_symbols(k) =  1 + 1j;  sym_id(k) = 1;
            elseif b1==0 && b2==1, qpsk_symbols(k) = -1 + 1j;  sym_id(k) = 2;
            elseif b1==1 && b2==1, qpsk_symbols(k) = -1 - 1j;  sym_id(k) = 3;
            else, qpsk_symbols(k) =  1 - 1j;  sym_id(k) = 4;
            end
        end
        qpsk_symbols = qpsk_symbols / sqrt(2);
        Sym{i} = sym_id;

        % Pulse shaping
        upsampled = upsample(qpsk_symbols, sps);
        bb = conv(upsampled, pulse, 'full');

        % 通過通道（非對稱）
        isi = conv(bb, h_total, 'full');

        % 修正總延遲，使符號對齊
        isi_aligned = isi(total_delay_samples+1 : total_delay_samples+signal_length);
        if length(isi_aligned) < signal_length
            isi_aligned = [isi_aligned; zeros(signal_length - length(isi_aligned), 1)];
        end
        
        % 根據模式確定 SNR
        if snr_is_range
            snr = snr_range(1) + (snr_range(2)-snr_range(1))*rand();
        else
            snr = snr_dB_fixed;
        end

        % 加入 AWGN
        noisy = EsN0_AWGN_corrected(snr, isi_aligned, K, sps);

        % 儲存結果
        Y(:,1,1,i) = single(real(isi_aligned));
        Y(:,1,2,i) = single(imag(isi_aligned));
        X(:,1,1,i) = single(real(noisy));
        X(:,1,2,i) = single(imag(noisy));
    end
end

% *** 輔助函數 2：AWGN 函數 (來自您的 CNN_LSTM 腳本) ***
function noisySignal = EsN0_AWGN_corrected(EsN0_dB, tx_signal, Ns, sps)
    Es_Total = sum(abs(tx_signal).^2);
    Es = Es_Total / Ns;
    EsN0_linear = 10^(EsN0_dB / 10);
    N0_per_symbol = Es / EsN0_linear;
    N0_per_sample = N0_per_symbol / sps;
    noise_variance_per_dim = N0_per_sample / 2;
    NoiseSigma = sqrt(noise_variance_per_dim);
    noise_I = NoiseSigma * randn(size(tx_signal));
    noise_Q = NoiseSigma * randn(size(tx_signal));
    noisySignal = tx_signal + (noise_I + 1j*noise_Q);
end


% clear; clc; close all;
% disp("開始執行 CNN vs DnCNN 在高SNR下的失真比較...");
% 
% % =========================================================================
% % --- 1. 訓練資料生成 (*** 關鍵修改 ***) ---
% % =========================================================================
% disp("正在生成 QPSK 訓練資料...");
% numTrainSamples = 20000; % 增加訓練樣本
% % *** 關鍵：使用寬廣的SNR範圍進行訓練，包含低SNR ***
% snr_dB_train_range = [-5, 20]; 
% fprintf("訓練SNR範圍: [%d, %d] dB\n", snr_dB_train_range(1), snr_dB_train_range(2));
% 
% frameHeight = 40;
% frameWidth = 40;
% % *** 關鍵：generateQPSKData 函數已被修改以接受 SNR 範圍 ***
% [XTrain, YTrain] = generateQPSKData(numTrainSamples, snr_dB_train_range, frameHeight, frameWidth);
% XTrain = single(XTrain);
% YTrain = single(YTrain);
% ZTrain = XTrain - YTrain; % ZTrain = trueNoise
% 
% % --- 2. 建立 Datastore ---
% dsX = arrayDatastore(XTrain, 'IterationDimension', 4);
% dsY = arrayDatastore(YTrain, 'IterationDimension', 4);
% dsZ = arrayDatastore(ZTrain, 'IterationDimension', 4);
% dsTrainX_Y = combine(dsX, dsY); % (noisy, clean)
% dsTrainX_Z = combine(dsX, dsZ); % (noisy, noise)
% disp("資料準備完成。");
% 
% % --- 3. DnCNN 網路定義 (高相容性版本) ---
% % (此架構將共用於兩個模型)
% layers = [
%     imageInputLayer([frameHeight frameWidth 2], Normalization="none", Name="input")
%     convolution2dLayer(3, 64, Padding="same", Name="conv1")
%     reluLayer(Name="relu1")
% ];
% % 中間層 (k=2 到 16)
% for k = 2:16
%     layers = [
%         layers
%         convolution2dLayer(3, 64, Padding="same", ...
%             BiasLearnRateFactor=0, Name="conv"+k)
%         batchNormalizationLayer("Name","bn"+k,"Epsilon",1e-4) 
%         reluLayer(Name="relu"+k)
%     ];
% end
% % 輸出層
% layers = [
%     layers
%     convolution2dLayer(3, 2, Padding="same", ...
%         BiasLearnRateFactor=0, Name="conv17")
% ];
% % 建立 LayerGraph
% lgraph = layerGraph(layers);
% % 轉換為 dlnetwork
% net = dlnetwork(lgraph);
% disp("網路架構(dlnetwork)建立完成。");
% 
% % --- 4. 訓練選項設定 ---
% opts = trainingOptions("adam", ...
%     MiniBatchSize=128, ...
%     MaxEpochs=50, ... % (為展示現象，20個epoch通常足夠)
%     InitialLearnRate=1e-3, ...
%     LearnRateSchedule="piecewise", ...
%     LearnRateDropPeriod=10, ... % 提早降低學習率
%     LearnRateDropFactor=0.5, ...
%     Shuffle="every-epoch", ...
%     Verbose=true, ...
%     ExecutionEnvironment="gpu", ... 
%     OutputNetwork="auto");
% 
% % --- 5. 訓練網路 (已修正) ---
% disp("開始訓練網路...");
% % --- 訓練模型一：標準 CNN (學習乾淨訊號 Y) ---
% disp("--- 正在訓練 標準 CNN (學習目標: Y_clean) ---");
% trainedNetX_Y = trainnet(dsTrainX_Y, net, "mse", opts);
% % --- 訓練模型二：DnCNN (學習噪聲 Z) ---
% disp("--- 正在訓練 DnCNN (學習目標: Z_noise) ---");
% trainedNetX_Z = trainnet(dsTrainX_Z, net, "mse", opts);
% disp("訓練完成！");
% 
% % =========================================================================
% % === 6. 訓練後效能驗證與繪圖 (*** 關鍵測試 ***) ===
% % =========================================================================
% disp("開始進行網路效能測試...");
% % 1. 生成新的測試資料
% numTestFrames = 5000;
% % *** 關鍵：使用極高 SNR 進行測試 ***
% snr_dB_test = 50; 
% fprintf("測試 SNR: %d dB (高信噪比測試)\n", snr_dB_test);
% [XTestNoisy, YTestClean] = generateQPSKData(numTestFrames, snr_dB_test, frameHeight, frameWidth);
% 
% % 2. 使用 trainedNet 進行預測 (去躁)
% dlXTestNoisy = dlarray(XTestNoisy, 'SSCB');
% dlYTestClean = dlarray(YTestClean, 'SSCB'); % 將乾淨資料也轉為 dlarray 以計算 MSE
% 
% % --- 執行模型一 (CNN) 預測 ---
% dlYPredClean = predict(trainedNetX_Y, dlXTestNoisy);
% % --- 執行模型二 (DnCNN) 預測 ---
% dlYPredNoise = predict(trainedNetX_Z, dlXTestNoisy);
% 
% % 從 GPU 和 dlarray 中取回資料
% YPredClean = extractdata(dlYPredClean); % CNN 的輸出 (預測的乾淨訊號)
% YPredNoise = extractdata(dlYPredNoise); % DnCNN 的輸出 (預測的噪聲)
% 
% % 3. 選擇一幀資料進行視覺化 (例如第 1 到 10 幀)
% i_frame = 1:10;
% 
% % --- 提取三種訊號的 I/Q 通道 ---
% noisy_I = XTestNoisy(:,:,1,i_frame);
% noisy_Q = XTestNoisy(:,:,2,i_frame);
% clean_I = YTestClean(:,:,1,i_frame);
% clean_Q = YTestClean(:,:,2,i_frame);
% 
% % --- 模型一 (CNN) 的去噪結果 ---
% denoised_I_CNN = YPredClean(:,:,1,i_frame);
% denoised_Q_CNN = YPredClean(:,:,2,i_frame);
% 
% % --- 模型二 (DnCNN) 的去噪結果 ---
% denoised_I_DnCNN = noisy_I - YPredNoise(:,:,1,i_frame);
% denoised_Q_DnCNN = noisy_Q - YPredNoise(:,:,2,i_frame);
% 
% % --- 轉換為 1D 複數序列 (共 16000 個符號) ---
% noisy_sym = noisy_I(:) + 1j * noisy_Q(:);
% clean_sym = clean_I(:) + 1j * clean_Q(:);
% denoised_sym_CNN = denoised_I_CNN(:) + 1j * denoised_Q_CNN(:);
% denoised_sym_DnCNN = denoised_I_DnCNN(:) + 1j * denoised_Q_DnCNN(:);
% 
% % --- 計算 MSE 進行定量比較 ---
% mse_CNN = mean(abs(denoised_sym_CNN - clean_sym).^2);
% mse_DnCNN = mean(abs(denoised_sym_DnCNN - clean_sym).^2);
% mse_Noisy = mean(abs(noisy_sym - clean_sym).^2);
% 
% fprintf("\n--- 高 SNR (%d dB) 測試結果 ---\n", snr_dB_test);
% fprintf("原始噪聲 MSE: %.6f\n", mse_Noisy);
% fprintf("標準 CNN 輸出 MSE: %.6f  (*** 預期此值較高, 代表失真 ***)\n", mse_CNN);
% fprintf("DnCNN 輸出 MSE: %.6f  (*** 預期此值接近噪聲MSE, 代表完美還原 ***)\n", mse_DnCNN);
% 
% % 4. 繪製星座圖 (Constellation Plot) - *** 2x2 比較圖 ***
% figure;
% set(gcf, 'Position', [50, 50, 1000, 800]); 
% 
% subplot(2,2,1);
% plot(noisy_sym, '.r'); % 紅色
% title(sprintf('Noisy Constellation (SNR = %d dB)', snr_dB_test));
% axis equal; grid on; xlim([-2 2]); ylim([-2 2]);
% xlabel('I-Channel'); ylabel('Q-Channel');
% 
% subplot(2,2,2);
% plot(clean_sym, '.k'); % 黑色
% title('Clean Constellation (Ground Truth)');
% axis equal; grid on; xlim([-2 2]); ylim([-2 2]);
% xlabel('I-Channel'); ylabel('Q-Channel');
% 
% subplot(2,2,3);
% plot(denoised_sym_CNN, '.b'); % 藍色
% title('Denoised (CNN)');
% axis equal; grid on; xlim([-2 2]); ylim([-2 2]);
% xlabel('I-Channel'); ylabel('Q-Channel');
% 
% subplot(2,2,4);
% plot(denoised_sym_DnCNN, '.g'); % 綠色
% title('Denoised (DnCNN)');
% axis equal; grid on; xlim([-2 2]); ylim([-2 2]);
% xlabel('I-Channel'); ylabel('Q-Channel');
% 
% sgtitle('CNN vs DnCNN 在高SNR下的失真比較', 'FontSize', 16, 'FontWeight', 'bold');
% 
% % 5. 繪製波形圖 (Waveform Plot) - *** 包含兩種模型 ***
% figure;
% set(gcf, 'Position', [50, 550, 1200, 600]);
% numSymToPlot = 50;
% sym_idx = 1:numSymToPlot;
% 
% % 繪製 I 通道
% subplot(2,1,1);
% plot(sym_idx, clean_I(sym_idx), 'k-', 'LineWidth', 2, 'DisplayName', 'Clean (I) (Ground Truth)');
% hold on;
% % plot(sym_idx, noisy_I(sym_idx), 'r:', 'DisplayName', 'Noisy (I)'); % 噪聲太小，可不畫
% plot(sym_idx, denoised_I_CNN(sym_idx), 'b--', 'LineWidth', 1.5, 'DisplayName', 'CNN (I) - 預期振幅衰減');
% plot(sym_idx, denoised_I_DnCNN(sym_idx), 'g-.', 'LineWidth', 1.5, 'DisplayName', 'DnCNN (I) - 預期完美重合');
% title('I-Channel Waveform (高SNR測試)');
% legend('Location', 'best'); grid on;
% ylabel('Amplitude');
% 
% % 繪製 Q 通道
% subplot(2,1,2);
% plot(sym_idx, clean_Q(sym_idx), 'k-', 'LineWidth', 2, 'DisplayName', 'Clean (Q) (Ground Truth)');
% hold on;
% % plot(sym_idx, noisy_Q(sym_idx), 'r:', 'DisplayName', 'Noisy (Q)');
% plot(sym_idx, denoised_Q_CNN(sym_idx), 'b--', 'LineWidth', 1.5, 'DisplayName', 'CNN (Q) - 預期振幅衰減');
% plot(sym_idx, denoised_Q_DnCNN(sym_idx), 'g-.', 'LineWidth', 1.5, 'DisplayName', 'DnCNN (Q) - 預期完美重合');
% title('Q-Channel Waveform (高SNR測試)');
% legend('Location', 'best'); grid on;
% xlabel('Symbol Index');
% ylabel('Amplitude');
% 
% sgtitle('CNN vs DnCNN 波形失真比較', 'FontSize', 16, 'FontWeight', 'bold');
% 
% disp('驗證繪圖完成！');
% 
% 
% % === 輔助函數 (Local Functions) ===
% 
% function [noisyData, cleanData] = generateQPSKData(numSamples, snr_input, H, W)
% % generateQPSKData - 生成 QPSK 訓練資料
% % *** 已修改：snr_input 可以是 1x2 範圍或純量 ***
% 
%     symbolsPerFrame = H * W; % 每幀的 QPSK 符號數 (1600)
%     M = 4; % QPSK
% 
%     % 預先配置記憶體 [H, W, C, B]
%     cleanData = zeros(H, W, 2, numSamples, 'single');
%     noisyData = zeros(H, W, 2, numSamples, 'single');
% 
%     % QPSK 符號 (Gray-coded)
%     constellation = pskmod(0:M-1, M, pi/4, 'gray');
% 
%     % 判斷 SNR 輸入是範圍還是純量
%     if isscalar(snr_input)
%         snr_is_range = false;
%         snr_dB_fixed = snr_input;
%     else
%         snr_is_range = true;
%         snr_range = snr_input;
%     end
% 
%     for i = 1:numSamples
%         % --- 生成乾淨訊號 ---
%         dataIn = randi([0 M-1], 1, symbolsPerFrame);
%         cleanSym = constellation(dataIn + 1);
% 
%         % --- 根據模式確定 SNR ---
%         if snr_is_range
%             % 隨機在範圍內生成 SNR
%             snr_dB = snr_range(1) + (snr_range(2) - snr_range(1)) * rand();
%         else
%             % 使用固定的 SNR
%             snr_dB = snr_dB_fixed;
%         end
% 
%         % --- 生成雜訊訊號 ---
%         noisySym = awgn(cleanSym, snr_dB, 'measured');
% 
%         % --- 格式化為 [H, W, C] ---
%         cleanReal = real(cleanSym);
%         noisyReal = real(noisySym);
%         cleanImag = imag(cleanSym);
%         noisyImag = imag(noisySym);
% 
%         % Channel 1: Real (I)
%         cleanData(:,:,1,i) = reshape(cleanReal, [H W]);
%         noisyData(:,:,1,i) = reshape(noisyReal, [H W]);
% 
%         % Channel 2: Imag (Q)
%         cleanData(:,:,2,i) = reshape(cleanImag, [H W]);
%         noisyData(:,:,2,i) = reshape(noisyImag, [H W]);
%     end
% end