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
        self.dropout33 = Dropout(p=0.1, inplace=False)

    def forward(self, x488):
        x489=self.dropout33(x488)
        return x489

m = M().eval()
x488 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x488)
end = time.time()
print(end-start)