import os
import numpy as np
import h5py
import librosa
import soundfile as sf
from scipy import interpolate
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Add these functions at the end of utils.py

def calculate_snr(signal_true, signal_pred):
    """Calculate Signal-to-Noise Ratio (SNR) in dB."""
    signal_power = np.mean(signal_true ** 2)
    noise_power = np.mean((signal_true - signal_pred) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def calculate_lsd(signal_true, signal_pred, n_fft=2048, hop_length=512):
    """Calculate Log-Spectral Distance (LSD)."""
    # Compute spectrograms
    stft_true = librosa.stft(signal_true, n_fft=n_fft, hop_length=hop_length)
    stft_pred = librosa.stft(signal_pred, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to magnitude spectra
    mag_true = np.abs(stft_true)
    mag_pred = np.abs(stft_pred)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    mag_true = np.maximum(mag_true, eps)
    mag_pred = np.maximum(mag_pred, eps)
    
    # Calculate LSD
    log_mag_true = np.log10(mag_true)
    log_mag_pred = np.log10(mag_pred)
    
    lsd = np.sqrt(np.mean((log_mag_true - log_mag_pred) ** 2))
    return lsd


def evaluate_audio_metrics(hr_audio, pr_audio):
    """Calculate evaluation metrics between high-res and predicted audio."""
    # Ensure same length
    min_len = min(len(hr_audio), len(pr_audio))
    hr_audio = hr_audio[:min_len]
    pr_audio = pr_audio[:min_len]
    
    metrics = {}
    
    # Time-domain metrics
    metrics['MSE'] = mean_squared_error(hr_audio, pr_audio)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(hr_audio, pr_audio)
    metrics['SNR_dB'] = calculate_snr(hr_audio, pr_audio)
    
    # Correlation
    if len(hr_audio) > 1:
        correlation, _ = pearsonr(hr_audio, pr_audio)
        metrics['Correlation'] = correlation
    else:
        metrics['Correlation'] = 0
    
    # Frequency-domain metrics
    metrics['LSD'] = calculate_lsd(hr_audio, pr_audio)
    
    return metrics

def load_h5(h5_path):
    """Load H5 dataset and return as PyTorch tensors."""
    with h5py.File(h5_path, 'r') as hf:
        print('List of arrays in input file:', list(hf.keys()))
        X = np.array(hf.get('data'))
        Y = np.array(hf.get('label'))
        print('Shape of X:', X.shape)
        print('Shape of Y:', Y.shape)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y


def spline_up(x_lr, r):
    """Spline upsampling using scipy interpolation."""
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp
    
# def calculate_snr(original, reconstructed):
#     """
#     Calculate Signal-to-Noise Ratio (SNR) in dB.
#     Higher values indicate better reconstruction quality.
#     """
#     # Convert to float and flatten
#     original = np.asarray(original, dtype=np.float64).flatten()
#     reconstructed = np.asarray(reconstructed, dtype=np.float64).flatten()

#     # Ensure same length
#     min_len = min(len(original), len(reconstructed))
#     original = original[:min_len]
#     reconstructed = reconstructed[:min_len]

#     # Compute noise
#     noise = original - reconstructed

#     # Use mean power for numerical stability
#     signal_power = np.mean(original ** 2)
#     noise_power = np.mean(noise ** 2)

#     # Handle perfect reconstruction case
#     if noise_power == 0:
#         return float('inf')

#     # Compute SNR in dB
#     snr = 10 * np.log10(signal_power / noise_power)
#     return snr

# def calculate_lsd(original, reconstructed, n_fft=2048, hop_length=512):
#     """
#     Calculate Log-Spectral Distance (LSD) in dB.
#     Lower is better.
#     """
#     # Compute STFT
#     S_orig = np.abs(librosa.stft(original, n_fft=n_fft, hop_length=hop_length))
#     S_recon = np.abs(librosa.stft(reconstructed, n_fft=n_fft, hop_length=hop_length))
    
#     # Convert to log scale
#     log_S_orig = np.log10(np.maximum(S_orig, 1e-8))
#     log_S_recon = np.log10(np.maximum(S_recon, 1e-8))
    
#     # Calculate LSD
#     lsd = np.sqrt(np.mean((log_S_orig - log_S_recon) ** 2))
#     return lsd

def upsample_wav(wav, args, model, save_spectrum=False):
    """Upsample a wav file using the trained model."""
    # Set model to evaluation mode
    model.eval()

    # Load signal
    x_hr, fs = librosa.load(wav, sr=args.sr)
    x_lr_t = np.array(x_hr[0::args.r])

    # Pad to multiple of patch size to ensure model runs over entire sample
    x_hr = np.pad(x_hr, (0, args.patch_size - (x_hr.shape[0] % args.patch_size)), 'constant', constant_values=(0, 0))

    # Downscale signal same as in data preparation
    x_lr = np.array(x_hr[0::args.r])

    # Upscale the low-res version
    x_lr = x_lr.reshape((1, len(x_lr), 1))

    # Preprocessing
    assert len(x_lr) == 1
    x_sp = spline_up(x_lr, args.r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(args.layers+1)))]
    x_sp = x_sp.reshape((1, len(x_sp), 1))
    x_sp = x_sp.reshape((int(x_sp.shape[1]/args.patch_size), args.patch_size, 1))

    # Convert to PyTorch tensor
    x_sp_tensor = torch.tensor(x_sp, dtype=torch.float32)

    # Move to device if GPU is available
    device = next(model.parameters()).device
    x_sp_tensor = x_sp_tensor.to(device)

    # Prediction
    with torch.no_grad():
        # Process in batches
        batch_size = 16
        pred_list = []

        for i in range(0, len(x_sp_tensor), batch_size):
            batch = x_sp_tensor[i:i+batch_size]
            pred_batch = model(batch)
            pred_list.append(pred_batch.cpu())

        pred = torch.cat(pred_list, dim=0)

    x_pr = pred.numpy().flatten()

    # Crop so that it works with scaling ratio
    x_hr = x_hr[:len(x_pr)]
    x_lr_t = x_lr_t[:len(x_pr)]

    # Save the files
    outname = wav  # + '.' + args.out_label
    sf.write(outname + '.lr.wav', x_lr_t, int(fs / args.r))
    sf.write(outname + '.hr.wav', x_hr, fs)
    sf.write(outname + '.pr.wav', x_pr, fs)

    # try:
    #     # Calculate metrics
    #     snr_value = calculate_snr(x_hr, x_pr)
    #     lsd_value = calculate_lsd(x_hr, x_pr)
        
    #     # Print metrics
    #     print(f"File: {os.path.basename(wav)}")
    #     print(f"  → SNR: {snr_value:.2f} dB")
    #     print(f"  → LSD: {lsd_value:.4f}")
    #     print("-" * 40)
    #     return snr_value, lsd_value
        
    # except Exception as e:
    #     print(f"Error calculating metrics for {wav}: {str(e)}")
    #     return None, None

    # if save_spectrum:
    #     # Save the spectrum
    #     S = get_spectrum(x_pr, n_fft=2048)
    #     save_spectrum(S, outfile=outname + '.pr.png')
    #     S = get_spectrum(x_hr, n_fft=2048)
    #     save_spectrum(S, outfile=outname + '.hr.png')
    #     S = get_spectrum(x_lr_t, n_fft=int(2048/args.r))
    #     save_spectrum(S, outfile=outname + '.lr.png')
    if evaluate_metrics:
        print(f"\nEvaluating metrics for: {os.path.basename(wav)}")
        metrics = evaluate_audio_metrics(x_hr, x_pr)
        print("-" * 50)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:<15}: {metric_value:8.4f}")
        print("-" * 50)
        
        # Return metrics for aggregation
        return metrics

    if save_spectrum:
        # Save the spectrum
        S = get_spectrum(x_pr, n_fft=2048)
        save_spectrum(S, outfile=outname + '.pr.png')
        S = get_spectrum(x_hr, n_fft=2048)
        save_spectrum(S, outfile=outname + '.hr.png')
        S = get_spectrum(x_lr_t, n_fft=int(2048/args.r))
        save_spectrum(S, outfile=outname + '.lr.png')

    return None

def get_spectrum(x, n_fft=2048):
    """Compute spectrum using STFT."""
    S = librosa.stft(x, n_fft=n_fft)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S


def save_spectrum(S, lim=800, outfile='spectrogram.png'):
    """Save spectrum plot."""
    plt.imshow(S.T, aspect=10)
    # plt.xlim([0,lim])
    plt.tight_layout()
    plt.savefig(outfile)
  

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f'Checkpoint saved: {filepath}')


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f'Checkpoint loaded: epoch {epoch}, loss {loss:.6f}')
    return epoch, loss

