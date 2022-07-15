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
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x71):
        x72=self.adaptiveavgpool2d0(x71)
        return x72

m = M().eval()
x71 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x71)
end = time.time()
print(end-start)
