import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.relu107 = ReLU()
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x375):
        x376=self.relu107(x375)
        x377=self.dropout1(x376)
        return x377

m = M().eval()
x375 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x375)
end = time.time()
print(end-start)
