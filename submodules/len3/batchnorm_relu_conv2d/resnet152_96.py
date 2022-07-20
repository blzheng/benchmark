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
        self.batchnorm2d149 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu145 = ReLU(inplace=True)
        self.conv2d150 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x493):
        x494=self.batchnorm2d149(x493)
        x495=self.relu145(x494)
        x496=self.conv2d150(x495)
        return x496

m = M().eval()
x493 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x493)
end = time.time()
print(end-start)
