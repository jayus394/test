import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnx.version_converter
import torch
import torch.onnx



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaitRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(GaitRecognitionModel, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 30, kernel_size=(100, 3), stride=(1, 3)),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )

        # Third convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=660, hidden_size=128, num_layers=2, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)

        # Softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Crop to 203*22*30
        x = x[:, :, :203, :22]

        # Reshape for LSTM input (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)

        # LSTM layer
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output from the last time step

        # Fully connected layer
        x = self.fc(x)

        # Softmax
        x = self.softmax(x)

        return x


# Load and preprocess data
def load_data(file_path):
    # Load .dat file and preprocess
    data = np.fromfile(file_path, dtype=np.float32).reshape(1, 1, 500, 90)
    data = torch.from_numpy(data).float()
    return data

data = torch.randn(1, 1, 500, 90)

# Initialize the model
num_classes = 10  # Adjust based on your needs
model = GaitRecognitionModel(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# Training loop (simplified)
def train(model, data, labels, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")




labels = torch.tensor([0])
num_epochs = 10

train(model, data, labels, num_epochs)


# Inference
def predict(model, data):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
    return predicted


torch.onnx.export(
    model,  # 要导出的模型
    data,  # 示例输入张量:
    "b.onnx",
    export_params=True,
)


model_file = 'p.onnx'
onnx_model = onnx.load(model_file)
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)




