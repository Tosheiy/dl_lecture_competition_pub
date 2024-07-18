import mne
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import gc
from sklearn.preprocessing import RobustScaler
from termcolor import cprint
from scipy.fftpack import fft

def pre_process(data, type):
    # TensorDatasetを作成
    dataset = TensorDataset(data)
    # DataLoaderを作成（バッチサイズを指定）
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    times = np.arange(281) / 100.0
    sfreq = 200  # オリジナルのサンプリングレート

    # 時間軸の生成
    sfreq = 200  # サンプリング周波数
    # (0秒からT秒までの時間配列)
    times = np.arange(data.shape[2]) / sfreq
    down = 1

    all_samples = [] 
    for batch_data in tqdm(dataloader, desc=f"{type}_PreProcess", leave=False):
        data = batch_data[0].to('cpu').detach().numpy().astype(np.float32).copy()

        # data = mne.filter.filter_data(data, sfreq, l_freq=1, h_freq=40) 
        # リサンプリング
        # 仮の200Hzデータを生成
        sampling_rate_original = 200  # 200Hz
        sampling_rate_new = 128  # 128Hz

        # MNEリサンプリング機能の使用
        data_resampled = mne.filter.resample(data, up=sampling_rate_new, down=sampling_rate_original)
        # data = mne.filter.resample(data, down=down) # 200Hz -> 100 Hz
        baseline_start = 0
        baseline_end = 20

        baseline_mean = np.mean(data[:, :, baseline_start:baseline_end], axis=2, keepdims=True)
        data = data - baseline_mean
        # ベースライン補正
        # data = mne.baseline.rescale(data, times, (0, 1.0), verbose=False) # バッチありでも行ける

        # # データの形状を取得
        # window_sizes = [100, 180, 240]
        # Batch_size, channels, time_seq = data.shape
        # current_pos = 0
        # # 補正後のデータを保存する配列を初期化
        # rescaled_data = np.copy(data)
        # for window_size in window_sizes:
        #     end_pos = min(current_pos + window_size, time_seq)
        #     baseline_data = data[:, :, current_pos:end_pos]
        #     # ウィンドウ内の平均を計算
        #     mean_baseline = np.mean(baseline_data, axis=2, keepdims=True)
        #     # ベースライン補正
        #     rescaled_data[:, :, current_pos:end_pos] -= mean_baseline
        #     current_pos = end_pos
        #     # 残り部分の補正
        #     if current_pos >= time_seq:
        #         break

        # # RobustScalerの初期化
        scaler = RobustScaler()

        # # 平均と標準偏差を計算
        # mean = np.mean(rescaled_data, axis=2, keepdims=True)  # 各チャネルごとに平均を計算
        # std = np.std(rescaled_data, axis=2, keepdims=True)    # 各チャネルごとに標準偏差を計算
        # # Zスコア標準化
        # data_normalized = (rescaled_data - mean) / std

        # 複素数で出力されるため却下
        # batch_size, channels, time_seq = data_normalized.shape
        # frequencies = np.fft.fftfreq(time_seq, d=1/sfreq)
        # fft_data = np.zeros((batch_size, channels, time_seq), dtype=np.complex64)
        # for i in range(batch_size):
        #     for j in range(channels):
        #         fft_data[i, j, :] = fft(data_normalized[i, j, :])

        # 各サンプル、各チャネルごとにスケーリング
        for i in range(data.shape[0]):
            data[i, :, :] = scaler.fit_transform(data[i, :, :])
        # 範囲外の値をクリッピング
        np.clip(data, -20, 20, out=data)


        all_samples.append(data.astype(np.float32))

        # メモリの開放
        del batch_data, data
        gc.collect()

    cprint(f"Pre process of {type} Finished.", "cyan")
    return torch.tensor(np.concatenate(all_samples, axis=0)).float()