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
        self.dropout32 = Dropout(p=0.0, inplace=False)
        self.linear35 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x405):
        x406=self.dropout32(x405)
        x407=self.linear35(x406)
        return x407

m = M().eval()
x405 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x405)
end = time.time()
print(end-start)
