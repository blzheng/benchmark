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
        self.relu79 = ReLU()

    def forward(self, x321):
        x322=self.relu79(x321)
        return x322

m = M().eval()
x321 = torch.randn(torch.Size([1, 144, 1, 1]))
start = time.time()
output = m(x321)
end = time.time()
print(end-start)
