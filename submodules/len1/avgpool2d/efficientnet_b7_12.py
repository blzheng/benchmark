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
        self.adaptiveavgpool2d12 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x183):
        x184=self.adaptiveavgpool2d12(x183)
        return x184

m = M().eval()
x183 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x183)
end = time.time()
print(end-start)
