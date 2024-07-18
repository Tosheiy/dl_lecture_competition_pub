import numpy as np
from sklearn.preprocessing import RobustScaler
import scipy.signal

def preprocess_meg(data, orig_sample_rate, new_sample_rate, tmin, tmax):
    
    
    # ステップ 3: ベースライン補正
    baseline_samples = int(abs(tmin) * new_sample_rate)
    for i in range(num_batches):
        baseline_mean = np.mean(epochs[i, :, :baseline_samples], axis=1, keepdims=True)
        epochs[i, :, :] -= baseline_mean
    
    # ステップ 4: ロバストスケーリング
    n_channels = epochs.shape[1]
    reshaped_data = epochs.transpose(0, 2, 1).reshape(-1, n_channels)
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(reshaped_data).reshape(epochs.shape[0], epochs.shape[2], n_channels).transpose(0, 2, 1)
    
    # ステップ 5: クリッピング
    np.clip(scaled_data, -20, 20, out=scaled_data)
    
    return scaled_data



