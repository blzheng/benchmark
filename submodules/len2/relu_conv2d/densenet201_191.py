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
        self.relu192 = ReLU(inplace=True)
        self.conv2d192 = Conv2d(1792, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x679):
        x680=self.relu192(x679)
        x681=self.conv2d192(x680)
        return x681

m = M().eval()
x679 = torch.randn(torch.Size([1, 1792, 7, 7]))
start = time.time()
output = m(x679)
end = time.time()
print(end-start)
