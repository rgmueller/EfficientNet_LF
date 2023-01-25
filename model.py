import torch
from torch import nn

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    print("Warning: GPU not available!")


def layer1_multistream(num_cams, filter_num):
    """

    :param num_cams:
    :param filter_num:
    :return:
    """
    if not hasattr(layer1_multistream, "instance"):
        layer1_multistream.instance = 0
    j = layer1_multistream.instance
    seq = nn.Sequential(nn.Conv2d(in_channels=num_cams,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.BatchNorm2d(num_features=filter_num),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.BatchNorm2d(num_features=filter_num),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.BatchNorm2d(num_features=filter_num),
                        nn.ReLU()
                        )
    layer1_multistream.instance += 1
    return seq

