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
        self.dropout15 = Dropout(p=0.1, inplace=False)

    def forward(self, x236):
        x237=self.dropout15(x236)
        return x237

m = M().eval()
x236 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x236)
end = time.time()
print(end-start)
