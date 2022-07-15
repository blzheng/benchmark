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
        self.adaptiveavgpool2d43 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x674):
        x675=self.adaptiveavgpool2d43(x674)
        return x675

m = M().eval()
x674 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x674)
end = time.time()
print(end-start)
