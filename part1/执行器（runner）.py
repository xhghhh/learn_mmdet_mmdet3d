# mmdet and mmdet3d,.etc are based on mmengine, and `runner` is really important in mmengine.

# get onto using mmengine within 15 minutes:

# 1. build the model
# 2. prepare the data and the data loader
# 3. the eval matrices
# 4. the runner

# 这里以在 CIFAR10 数据集上训练 ResNet50 为例，展示如何使用 mmengine 进行模型训练、验证和测试。

import torch.nn.functional as F
import torchvision
import mmengine
from mmengine.model import BaseModel
print(mmengine.__version__)



# 1. build the model
# 我们约定，所有的模型都继承自 `BaseModel` 类，并实现 `forward` 方法，它接受数据集的参数、额外参数`mode`。

# for training, mode 接受字符串 'loss'，返回一个包含 loss 字段的字典。
# 对验证集和测试集，mode 接受字符串 'predict'，返回预测结果和真实信息。

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':  # training
            return {'loss': F.cross_entropy(x, labels)}  # 交叉熵损失
        elif mode == 'predict':  # validation or testing
            return x, labels  # 返回预测结果和真实信息

# 2. prepare the data and the data loader
# 构建数据集和数据加载器
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 这里我们使用 CIFAR10 数据集
# 这里的 dataset 和 dataloader 是 torch.utils.data.Dataset 和 torch.utils.data.DataLoader 的子类
# 也就是符合 pytorch 的数据集格式
norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))

# 3. the eval standard
# 评价指标
# 我们约定这一评测指标需要继承 BaseMetric，并实现 process 和 compute_metrics 方法
# 其中 process 方法接受数据集的输出和模型 mode="predict" 时的输出，此时的数据为一个批次的数据，
# 对这一批次的数据进行处理后，保存信息至 self.results 属性

from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(accuracy=100 * total_correct / total_size)

# 4. the runner

from torch.optim import SGD
from mmengine.runner import Runner

runner = Runner(
    # 用以训练和验证的模型，需要满足特定的接口需求
    model=MMResNet50(),
    # 工作路径，用以保存训练日志、权重文件信息
    work_dir='./work_dir',
    # 训练数据加载器，需要满足 PyTorch 数据加载器协议
    train_dataloader=train_dataloader,
    # 优化器包装，用于模型优化，并提供 AMP、梯度累积等附加功能
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # 训练配置，用于指定训练周期、验证间隔等信息
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    # 验证数据加载器，需要满足 PyTorch 数据加载器协议
    val_dataloader=val_dataloader,
    # 验证配置，用于指定验证所需要的额外参数
    val_cfg=dict(),
    # 用于验证的评测器，这里使用默认评测器，并评测指标
    val_evaluator=dict(type=Accuracy),
)

runner.train()