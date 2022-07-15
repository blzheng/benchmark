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
        self.adaptiveavgpool2d27 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x497):
        x498=self.adaptiveavgpool2d27(x497)
        return x498

m = M().eval()
x497 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x497)
end = time.time()
print(end-start)
