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
        self.relu5 = ReLU(inplace=True)
        self.dropout1 = Dropout(p=0.5, inplace=False)
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)

    def forward(self, x17):
        x18=self.relu5(x17)
        x19=self.dropout1(x18)
        x20=self.linear1(x19)
        return x20

m = M().eval()
x17 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x17)
end = time.time()
print(end-start)
