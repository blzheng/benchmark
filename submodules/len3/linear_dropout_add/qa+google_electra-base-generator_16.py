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
        self.linear52 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout26 = Dropout(p=0.1, inplace=False)

    def forward(self, x397, x365):
        x398=self.linear52(x397)
        x399=self.dropout26(x398)
        x400=operator.add(x399, x365)
        return x400

m = M().eval()
x397 = torch.randn(torch.Size([1, 384, 256]))
x365 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x397, x365)
end = time.time()
print(end-start)
