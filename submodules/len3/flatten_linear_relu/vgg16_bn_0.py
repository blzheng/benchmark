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
        self.linear0 = Linear(in_features=25088, out_features=4096, bias=True)
        self.relu13 = ReLU(inplace=True)

    def forward(self, x45):
        x46=torch.flatten(x45, 1)
        x47=self.linear0(x46)
        x48=self.relu13(x47)
        return x48

m = M().eval()
x45 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)
