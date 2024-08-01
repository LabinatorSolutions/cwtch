#
# @chatgpt @claude read these comments and don't delete them when changing the code
# chess nnue trainer
# 768 sparse binary inputs (max 32 1 values)
# 768 x HIDDEN_SIZE (relu) x 1 (sigmoid) model
# sigmoid needs its input divided by SIGMOID_STRETCH
# streamed BATCH_SIZE batched data file
# average batch loss updated during epoch
#

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import json
import sys

NUM_WORKERS = 6
HIDDEN_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000
LOG_FREQUENCY = 100
SIGMOID_STRETCH = 100
WEIGHTS_FILE = 'data/weights.js'
DATA_FILE = 'data/ql3.epd'

def fen_to_input(fen_line):
    # Split the FEN line into position and result
    fen, result = fen_line.rsplit(' ', 1)

    # Extract the board position part of the FEN
    board_fen = fen.split(' ')[0]

    # Initialize the 768-element input list
    input_list = [0] * 768

    # Define piece to index mapping
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5  # Black pieces use same index, but we'll add 384 later
    }

    # Parse the board FEN
    square_index = 0
    for char in board_fen:
        if char == '/':
            continue
        elif char.isdigit():
            square_index += int(char)
        else:
            piece_index = piece_to_index[char]
            if char.isupper():  # White piece
                input_index = piece_index * 64 + square_index
            else:  # Black piece
                input_index = 384 + piece_index * 64 + square_index
            input_list[input_index] = 1
            square_index += 1

    # Convert input list to PyTorch tensor
    input_tensor = torch.tensor(input_list, dtype=torch.float32)

    # Convert result to target value
    result_to_target = {'w': 1.0, 'd': 0.5, 'b': 0.0}
    if result not in result_to_target:
        print(f"Unexpected result value: {result}")
        sys.exit(1)
    target = result_to_target[result]
    target_tensor = torch.tensor([target], dtype=torch.float32)

    return input_tensor, target_tensor

class ChessDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(self.file_path, 'r') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        input_tensor, target_tensor = fen_to_input(line)
        return input_tensor, target_tensor

class ChessNNUE(nn.Module):
    def __init__(self):
        super(ChessNNUE, self).__init__()
        self.fc1 = nn.Linear(768, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x) / SIGMOID_STRETCH
        x = self.sigmoid(x)
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        #init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)

def save_model_weights(model, epoch, loss):
    with open(WEIGHTS_FILE, 'w') as f:
        f.write("{{{  weights\n")
        f.write(f"const net_h1_size    = {HIDDEN_SIZE};\n")
        f.write(f"const net_batch_size = {BATCH_SIZE};\n")
        f.write(f"const net_activation = 'relu';\n")
        f.write(f"const net_stretch    = {SIGMOID_STRETCH};\n")
        f.write(f"const net_epochs     = {epoch};\n")
        f.write(f"const net_loss       = {loss};\n")
        f.write("const net_h1_w = Array(768);\n")
        f.write("{{{  weights\n")
        for i in range(model.fc1.weight.size(1)):  # size(1) gives the number of input features (768)
            weights = model.fc1.weight[:, i].tolist()
            f.write(f"net_h1_w[{i}] = new Float32Array({weights});\n")
        biases_h1 = model.fc1.bias.tolist()
        f.write(f"net_h1_b = new Float32Array({biases_h1});\n")
        weights_fc2 = model.fc2.weight.squeeze().tolist()
        f.write(f"net_o_w = new Float32Array({weights_fc2});\n")
        bias_fc2 = model.fc2.bias.item()
        f.write(f"net_o_b = {bias_fc2};\n")
        f.write("}}}\n")
        f.write("}}}\n")

def train(model, dataloader, criterion, optimizer, num_epochs=NUM_EPOCHS, log_frequency=LOG_FREQUENCY):
    model.train()
    running_loss = 0;
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % log_frequency == 0 or i == len(dataloader)-1:
                avg_loss = running_loss / (i+1)
                print(f"\repoch [{epoch+1}/{num_epochs}], batch [{i+1}/{len(dataloader)}], loss: {avg_loss:.8f}", end='')
        print()
        save_model_weights(model, epoch+1, running_loss)

def main():
    dataset = ChessDataset(DATA_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    model = ChessNNUE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    save_model_weights(model, 0, 0)
    train(model, dataloader, criterion, optimizer)

main()

