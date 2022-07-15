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
        self.adaptiveavgpool2d36 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x563):
        x564=self.adaptiveavgpool2d36(x563)
        return x564

m = M().eval()
x563 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x563)
end = time.time()
print(end-start)
