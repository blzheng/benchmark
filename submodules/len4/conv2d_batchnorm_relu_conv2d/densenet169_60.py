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
        self.conv2d124 = Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d125 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu125 = ReLU(inplace=True)
        self.conv2d125 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x442):
        x443=self.conv2d124(x442)
        x444=self.batchnorm2d125(x443)
        x445=self.relu125(x444)
        x446=self.conv2d125(x445)
        return x446

m = M().eval()
x442 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x442)
end = time.time()
print(end-start)
