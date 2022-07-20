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
        self.conv2d36 = Conv2d(384, 768, kernel_size=(2, 2), stride=(2, 2))
        self.conv2d37 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)

    def forward(self, x386):
        x387=self.conv2d36(x386)
        x389=self.conv2d37(x387)
        return x389

m = M().eval()
x386 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x386)
end = time.time()
print(end-start)
