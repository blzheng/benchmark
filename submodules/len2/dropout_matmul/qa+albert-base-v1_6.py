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
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x275, x269):
        x276=self.dropout1(x275)
        x277=torch.matmul(x276, x269)
        return x277

m = M().eval()
x275 = torch.randn(torch.Size([1, 12, 384, 384]))
x269 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x275, x269)
end = time.time()
print(end-start)