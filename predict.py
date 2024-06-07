import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
from cnn import CNNNetwork
from train import prepare_data


def load_model(model_path, num_classes, device):
    model = CNNNetwork(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, device, data_loader, label_map):
    predictions = []
    model.eval()
    with torch.no_grad():
        for signals in data_loader:
            signals = signals.to(device)
            outputs = model(signals)
            # print(outputs)
            # _, predicted = torch.max(outputs, 1)
            # print(predicted)
            predictions.extend(outputs.cpu().numpy())
            # print(predictions)

    # inverse_label_map = {v: k for k, v in label_map.items()}
    # print(inverse_label_map)

    # predicted_labels = [inverse_label_map[pred] for pred in predictions]
    # return predicted_labels
    return predictions


def predict_folder(model_path, folder_path, label_map, device, sample_rate, num_samples, batch_size=32):
    w = 0
    inverse_label_map = {v: k for k, v in label_map.items()}
    file_paths = []
    first_write = True
    for f in os.listdir(folder_path):
        if f.endswith('.ogg'):
            file_paths.append(os.path.join(folder_path, f))
            w += 1

            if (w % 31 == 0 and w > 0) or w >= 1000:
                print('Batch: ', w, '. ')
                data = [prepare_data(fp, device, sample_rate, num_samples) for fp in file_paths]
                data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

                num_classes = len(label_map)
                model = load_model(model_path, num_classes, device)
                predicted_probs = predict(model, device, data_loader, label_map)

                predictions = []

                for file_path, probs in zip(file_paths, predicted_probs):
                    # print(f"File: {file_path})
                    file_name = file_path.split("/")[-1]
                    file_name = file_name.split(".")[-2]
                    # print(file_name, probs)
                    predictions.append(
                        {'row_id': file_name + '_240', **{inverse_label_map[i]: prob for i, prob in enumerate(probs)}})
                # Convert the list to a DataFrame and save it as a CSV file
                df = pd.DataFrame(predictions)
                if first_write:
                    df.to_csv('submission.csv', index=False)
                    first_write = False
                else:
                    df.to_csv('submission.csv', mode='a', header=False, index=False)
                file_paths = []
