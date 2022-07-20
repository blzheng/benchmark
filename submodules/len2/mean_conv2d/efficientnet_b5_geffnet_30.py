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
        self.conv2d150 = Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x447):
        x448=x447.mean((2, 3),keepdim=True)
        x449=self.conv2d150(x448)
        return x449

m = M().eval()
x447 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x447)
end = time.time()
print(end-start)
