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
        self.conv2d30 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)

    def forward(self, x314, x304):
        x315=operator.add(x314, x304)
        x317=self.conv2d30(x315)
        return x317

m = M().eval()
x314 = torch.randn(torch.Size([1, 768, 14, 14]))
x304 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x314, x304)
end = time.time()
print(end-start)
