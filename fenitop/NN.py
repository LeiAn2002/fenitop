import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import StepLR


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(NeuralNetwork, self).__init__()
  
        self.input_layer = nn.Linear(input_size, hidden1_size)
        self.hidden_layer1 = nn.Linear(hidden1_size, hidden2_size)
        self.hidden_layer2 = nn.Linear(hidden2_size, output_size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):

        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer1(x))
        x = self.hidden_layer2(x)
        return x

if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)

    data = np.load('incremental_training_data.npz')
    input_data = data['inputs']
    labels = data['labels']
    input_data = input_data.reshape(680, 4)

    input_data = torch.tensor(input_data, dtype=torch.float64) 
    labels = torch.tensor(labels, dtype=torch.float64)          

    dataset = TensorDataset(input_data, labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    input_size = 4 
    hidden1_size = 256
    hidden2_size = 256
    output_size = 6 

    model = NeuralNetwork(input_size, hidden1_size, hidden2_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    num_epochs = 200

    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, targets)  
            loss.backward()
            optimizer.step() 
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

    model.eval() 
    test_loss = 0.0

    with torch.no_grad(): 
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved to trained_model.pth")
