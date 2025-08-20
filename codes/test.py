from models.afilm import AFiLMModel
from models.tfilm import TFiLMModel
from utils import upsample_wav
import os
import argparse
import torch

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', required=True,
        help='path to pre-trained model')
    parser.add_argument('--out-label', default='',
        help='append label to output samples')
    parser.add_argument('--wav-file-list',
        help='list of audio files for evaluation')
    parser.add_argument('--layers', default=4, type=int,
            help='number of layers in each of the D and U halves of the network')
    parser.add_argument('--r', help='upscaling factor', default=4, type=int)
    parser.add_argument('--sr', help='high-res sampling rate',
                                    type=int, default=16000)
    parser.add_argument('--patch_size', type=int, default=8192,
                        help='Size of patches over which the model operates')
    parser.add_argument('--device', default='auto',
                        help='device to use (auto, cpu, cuda)')
    return parser


def get_device(device_arg):
    """Get the appropriate device based on argument and availability."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        device = torch.device('cuda')
    elif device_arg == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(device_arg)

    print(f'Using device: {device}')
    return device


def load_model(model_path, device, n_layers=4, scale=4):
    """Load pre-trained model."""
    print(f"Loading model from {model_path}")

    # Create model instance
    from models.afilm import get_afilm
    model = get_afilm(n_layers=n_layers, scale=scale)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint and 'loss' in checkpoint:
            print(f"Loaded model from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f}")
    else:
        # Assume the file contains just the state dict
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def test(args):
    # Set device
    device = get_device(args.device)

    # Load model
    model = load_model(args.pretrained_model, device, n_layers=args.layers, scale=args.r)

    # Process audio files
    if args.wav_file_list:
        # If it's a single file
        if args.wav_file_list.endswith('.wav'):
            print(f"Processing single file: {args.wav_file_list}")
            upsample_wav(args.wav_file_list, args, model)
        else:
            # If it's a file containing list of wav files
            print(f"Processing file list: {args.wav_file_list}")
            with open(args.wav_file_list, 'r') as f:
                wav_files = [line.strip() for line in f.readlines() if line.strip()]

            for i, wav_file in enumerate(wav_files):
                print(f"Processing file {i+1}/{len(wav_files)}: {wav_file}")
                try:
                    upsample_wav(wav_file, args, model)
                    print(f"Successfully processed: {wav_file}")
                except Exception as e:
                    print(f"Error processing {wav_file}: {str(e)}")
    else:
        print("No wav file list provided. Use --wav-file-list to specify files to process.")

def main():
    parser = make_parser()
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
