from predict import predict_folder
from train import data_loader
import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    folder_path = '/kaggle/input/birdclef-2024/train_audio'
    label_map = {}

    # Prepare data
    data = data_loader(device, SAMPLE_RATE, NUM_SAMPLES, folder_path, label_map)

    FOLDER_PATH = '/kaggle/input/birdclef-2024/unlabeled_soundscapes'
    MODEL_PATH = '/kaggle/working/train_result.pth'

    predict_folder(MODEL_PATH, FOLDER_PATH, label_map, device, SAMPLE_RATE, NUM_SAMPLES)