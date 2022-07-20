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
        self.linear60 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu30 = GELU(approximate='none')
        self.linear61 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x352):
        x353=self.linear60(x352)
        x354=self.gelu30(x353)
        x355=self.linear61(x354)
        return x355

m = M().eval()
x352 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x352)
end = time.time()
print(end-start)
