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
        self.linear23 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout21 = Dropout(p=0.0, inplace=False)

    def forward(self, x268):
        x269=self.linear23(x268)
        x270=self.dropout21(x269)
        return x270

m = M().eval()
x268 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x268)
end = time.time()
print(end-start)
