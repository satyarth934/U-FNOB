import logging
from torch import nn


class PrintShape(nn.Module):
    def __init__(
        self, 
        msg,
        logger_level=logging.DEBUG,
    ):
        super(PrintShape, self).__init__()
        self.msg = msg

        # Defining logger
        self.logger_ = logging.getLogger(self.__class__.__name__)
        self.logger_.setLevel(logger_level)

    def forward(self, x):
        self.logger_.debug(f"{self.msg}: {x.shape = }")
        return x