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
        self.adaptiveavgpool2d38 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x594):
        x595=self.adaptiveavgpool2d38(x594)
        return x595

m = M().eval()
x594 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x594)
end = time.time()
print(end-start)
