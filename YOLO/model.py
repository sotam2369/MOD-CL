import torch.nn as nn

class MaxSAT_Model(nn.Module):
    def __init__(self, input=41):
        super(MaxSAT_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 41)
        )
        

    def forward(self, x):
        return self.model(x)


class ExtenderModel(nn.Module):
    def __init__(self, input=41):
        super(ExtenderModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 64),
            nn.ReLU(),
            nn.Linear(64, 41)
        )
        

    def forward(self, x):
        return self.model(x)



class CombinerModel(nn.Module):
    def __init__(self, input=82):
        super(CombinerModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 64),
            nn.ReLU(),
            nn.Linear(64, 41)
        )
        

    def forward(self, x):
        return self.model(x)