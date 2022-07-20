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
        self.relu53 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(784, 784, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=49, bias=False)

    def forward(self, x221):
        x222=self.relu53(x221)
        x223=self.conv2d71(x222)
        return x223

m = M().eval()
x221 = torch.randn(torch.Size([1, 784, 7, 7]))
start = time.time()
output = m(x221)
end = time.time()
print(end-start)
