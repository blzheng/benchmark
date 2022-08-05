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

    def forward(self, x277, x275):
        x278=self.linear36(x277)
        x279=self.dropout18(x278)
        x280=operator.add(x279, x275)
        return x280

m = M().eval()
x277 = torch.randn(torch.Size([1, 384, 1024]))
x275 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x277, x275)
end = time.time()
print(end-start)
