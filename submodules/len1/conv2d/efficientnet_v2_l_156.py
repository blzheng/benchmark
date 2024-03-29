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
        self.conv2d156 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x506):
        x507=self.conv2d156(x506)
        return x507

m = M().eval()
x506 = torch.randn(torch.Size([1, 56, 1, 1]))
start = time.time()
output = m(x506)
end = time.time()
print(end-start)
