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
        self.dropout29 = Dropout(p=0.0, inplace=False)

    def forward(self, x361):
        x362=self.dropout29(x361)
        return x362

m = M().eval()
x361 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x361)
end = time.time()
print(end-start)
