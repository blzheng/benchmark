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
        self.adaptiveavgpool2d45 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x705):
        x706=self.adaptiveavgpool2d45(x705)
        return x706

m = M().eval()
x705 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x705)
end = time.time()
print(end-start)