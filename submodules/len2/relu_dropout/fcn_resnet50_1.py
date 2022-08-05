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
        self.relu50 = ReLU()
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x182):
        x183=self.relu50(x182)
        x184=self.dropout1(x183)
        return x184

m = M().eval()
x182 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)
