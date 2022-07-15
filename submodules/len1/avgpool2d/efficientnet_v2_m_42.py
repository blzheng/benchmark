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
        self.adaptiveavgpool2d42 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x754):
        x755=self.adaptiveavgpool2d42(x754)
        return x755

m = M().eval()
x754 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x754)
end = time.time()
print(end-start)
