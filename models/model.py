import torch
import torch.nn as nn

class DNAMultitaskModel(nn.Module):
    def __init__(self):
      super(DNAMultitaskModel, self).__init__()

      self.conv1 = nn.Conv1d(4, 64, kernel_size=21, padding='same')
      self.relu1 = nn.ReLU()
      self.pool1 = nn.MaxPool1d(2)
      self.drop1 = nn.Dropout1d(0.1)

      self.conv2 = nn.Conv1d(64,128,11, padding='same')
      self.relu2 = nn.ReLU()
      self.pool2 = nn.MaxPool1d(2)
      self.drop2 = nn.Dropout1d(0.2)


      self.global_pool = nn.AdaptiveAvgPool1d(1)

      self.fc_class = nn.Linear(128,1)
      self.sigmoid = nn.Sigmoid()

      self.fc_reg = nn.Linear(128,1)

    def forward(self, x):

      x = self.conv1(x)
      x = self.relu1(x)
      x = self.pool1(x)
      x = self.drop1(x)

      x = self.conv2(x)
      x = self.relu2(x)
      x = self.pool2(x)
      x = self.drop2(x)


      x = self.global_pool(x)
      x = torch.flatten(x, 1)

      out_class = self.fc_class(x)
      out_class = self.sigmoid(out_class)

      out_reg = self.fc_reg(x)

      return out_class, out_reg
