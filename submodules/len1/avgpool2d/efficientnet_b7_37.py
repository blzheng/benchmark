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
        self.adaptiveavgpool2d37 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x579):
        x580=self.adaptiveavgpool2d37(x579)
        return x580

m = M().eval()
x579 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x579)
end = time.time()
print(end-start)
