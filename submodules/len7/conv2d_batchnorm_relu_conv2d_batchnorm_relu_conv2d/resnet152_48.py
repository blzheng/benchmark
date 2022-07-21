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
        self.conv2d149 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d149 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu145 = ReLU(inplace=True)
        self.conv2d150 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d150 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d151 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x492):
        x493=self.conv2d149(x492)
        x494=self.batchnorm2d149(x493)
        x495=self.relu145(x494)
        x496=self.conv2d150(x495)
        x497=self.batchnorm2d150(x496)
        x498=self.relu145(x497)
        x499=self.conv2d151(x498)
        return x499

m = M().eval()
x492 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x492)
end = time.time()
print(end-start)
