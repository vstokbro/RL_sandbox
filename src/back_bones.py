import torch
import torch.nn as nn

# Define the neural network architecture
class BaseModel(nn.Module):
    def __init__(self,input_featues,output_features,hidden_units):
        super(BaseModel, self).__init__()
        self.input_features = input_featues
        self.fc1 = nn.Linear(input_featues, hidden_units[0])  # input layer -> hidden layer
        nn.init.kaiming_uniform_(self.fc1.weight,mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])   # hidden layer -> output layer
        nn.init.kaiming_uniform_(self.fc2.weight,mode='fan_in', nonlinearity='relu')
        self.fc3 = nn.Linear(hidden_units[1], output_features)

    def forward(self, x):
        x = x.view(-1, self.input_features).float()  # flatten the input image
        x = torch.relu(self.fc1(x))  # pass through the first hidden layer
        x = torch.relu(self.fc2(x))  # pass through the output layer
        return self.fc3(x)
    
class NormDistNet(BaseModel):
    def __init__(self, input_featues, output_features, hidden_units,std_grad,std_init):
        super().__init__(input_featues, output_features, hidden_units)
        log_std = torch.log(torch.ones(output_features)*std_init)
        self.log_std =torch.nn.Parameter(log_std,requires_grad=bool(std_grad))
    
    def forward(self, x):
        return torch.tanh(super().forward(x)), self.log_std
