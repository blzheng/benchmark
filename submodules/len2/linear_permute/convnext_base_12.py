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
        self.linear25 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x156):
        x157=self.linear25(x156)
        x158=torch.permute(x157, [0, 3, 1, 2])
        return x158

m = M().eval()
x156 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
