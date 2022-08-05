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
        self.gelu15 = GELU(approximate='none')
        self.dropout30 = Dropout(p=0.0, inplace=False)
        self.linear33 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout31 = Dropout(p=0.0, inplace=False)

    def forward(self, x381):
        x382=self.gelu15(x381)
        x383=self.dropout30(x382)
        x384=self.linear33(x383)
        x385=self.dropout31(x384)
        return x385

m = M().eval()
x381 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x381)
end = time.time()
print(end-start)
