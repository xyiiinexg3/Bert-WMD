import torch
class FCModel(torch.nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc = torch.nn.Linear(in_features=768, out_features=1)

    def forward(self, input):
        score = self.fc(input)
        result = torch.sigmoid(score)
        return result