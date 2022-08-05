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
        self.relu54 = ReLU()
        self.dropout0 = Dropout(p=0.5, inplace=False)

    def forward(self, x196):
        x197=self.relu54(x196)
        x198=self.dropout0(x197)
        return x198

m = M().eval()
x196 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x196)
end = time.time()
print(end-start)
