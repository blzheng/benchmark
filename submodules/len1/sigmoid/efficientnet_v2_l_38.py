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
        self.sigmoid38 = Sigmoid()

    def forward(self, x729):
        x730=self.sigmoid38(x729)
        return x730

m = M().eval()
x729 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x729)
end = time.time()
print(end-start)
