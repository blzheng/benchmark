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
        self.linear13 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout11 = Dropout(p=0.0, inplace=False)

    def forward(self, x153):
        x154=self.linear13(x153)
        x155=self.dropout11(x154)
        return x155

m = M().eval()
x153 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
