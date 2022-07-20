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
        self.relu11 = ReLU(inplace=True)

    def forward(self, x46, x41):
        x47=operator.add(x46, x41)
        x48=self.relu11(x47)
        return x48

m = M().eval()
x46 = torch.randn(torch.Size([1, 128, 28, 28]))
x41 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x46, x41)
end = time.time()
print(end-start)
