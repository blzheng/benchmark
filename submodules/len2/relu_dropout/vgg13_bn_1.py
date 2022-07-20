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
        self.dropout1 = Dropout(p=0.5, inplace=False)

    def forward(self, x41):
        x42=self.relu11(x41)
        x43=self.dropout1(x42)
        return x43

m = M().eval()
x41 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x41)
end = time.time()
print(end-start)
