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
        self.linear36 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout18 = Dropout(p=0.1, inplace=False)

    def forward(self, x276):
        x277=torch._C._nn.gelu(x276)
        x278=self.linear36(x277)
        x279=self.dropout18(x278)
        return x279

m = M().eval()
x276 = torch.randn(torch.Size([1, 384, 1024]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
