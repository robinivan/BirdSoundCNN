import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from cnn import CNNNetwork
from torch import nn, optim


def prepare_data(path, device, target_sample_rate, num_samples):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    transform = mel_spectrogram.to(device)

    signal, sr = torchaudio.load(path)
    signal = signal.to(device)
    signal = resample_if_necessary(signal, sr, target_sample_rate)
    signal = mix_down_if_necessary(signal)
    signal = cut_if_necessary(signal, num_samples)
    signal = right_pad_if_necessary(signal, num_samples)
    # signal = add_noise(signal, np.random())
    signal = transform(signal)
    return signal


def resample_if_necessary(signal, sr, target_sample_rate):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal


def mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def cut_if_necessary(signal, num_samples):
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal


def right_pad_if_necessary(signal, num_samples):
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal


def data_loader(device, sample_rate, num_samples, folder_path, label_map):
    data = []
    i = 0
    for dirname, _, filenames in os.walk(folder_path):
        last_folder = dirname.split("/")[-1]
        k = 0
        if last_folder != 'train_audio':
            if last_folder not in label_map:
                label_map[last_folder] = len(label_map)
            label = label_map[last_folder]
            print(i, ' start extract: ', last_folder)
            for filename in filenames:
                signal = prepare_data(os.path.join(dirname, filename), device, sample_rate, num_samples)
                data.append((signal, label))
                k += 1
            print('Extracted files: ', k)
            # print(data)
            # training starts
            if i % 5 == 0 or i == 182:
                # Split data into training and testing sets
                train_size = int(0.8 * len(data))
                test_size = len(data) - train_size
                train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

                # Create data loaders
                train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

                # Initialize model, loss function, and optimizer
                num_classes = len(label_map)  # Get number of classes from label map
                cnn = CNNNetwork(num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(cnn.parameters(), lr=0.001)

                # Train the model
                train(cnn, device, train_loader, criterion, optimizer, num_epochs=10)

                # Evaluate the model
                evaluate(cnn, device, test_loader)

                # Save the model
                torch.save(cnn.state_dict(), '/kaggle/working/train_result.pth')
                # training ends
                data = []
        i += 1
    return data


def train(model, device, train_loader, criterion, optimizer, num_epochs):
    model.to(device)
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device).long()  # Ensure labels are of type long
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    print('Finished Training')


def evaluate(model, device, test_loader):
    model.to(device)
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')


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
