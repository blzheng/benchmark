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
        self.conv2d15 = Conv2d(244, 244, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=244, bias=False)

    def forward(self, x94):
        x95=self.conv2d15(x94)
        return x95

m = M().eval()
x94 = torch.randn(torch.Size([1, 244, 28, 28]))
start = time.time()
output = m(x94)
end = time.time()
print(end-start)
