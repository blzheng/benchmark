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
        self.conv2d75 = Conv2d(720, 720, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=720, bias=False)

    def forward(self, x230):
        x231=self.conv2d75(x230)
        return x231

m = M().eval()
x230 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x230)
end = time.time()
print(end-start)
