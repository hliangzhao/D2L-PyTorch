from IPython import display
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


def use_svg_display():
    """
    使用矢量图显示图表。
    """
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """
    设置图表大小。
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def mgd(params, lr, batch_size):
    """
    Mini-batch Gradient Desecnt.
    """
    for param in params:
        param.data -= lr * param.grad / batch_size


def linreg(X, w, b):
    """
    定义线性回归模型。
    """
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    """
    定义均方误差损失函数。
    """
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def get_fashion_mnist_labels(labels):
    """
    根据数字获得对应的字符串label。
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    """
    展示给定的图片及对应的label。
    """
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(10, 10))
    # 一张img和对应的lbl填充一个fig
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_fashion_mnist(batch_size):
    """
    将fashion-MNIST数据集导入变量fashion_mnist_train和fashion_mnist_test。
    """
    fashion_mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    fashion_mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()
    )
    num_workers = 4    # 开启4个线程
    train_iter = torch.utils.data.DataLoader(
        fashion_mnist_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        fashion_mnist_test, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    return train_iter, test_iter


# below code is only suitable for .ipynb before 03-7
# def evaluate_accuracy(data_iter, net):
#     """
#     计算mini-batch数据上给定的模型的正确率。
#     """
#     acc_sum, n = 0., 0
#     for X, y in data_iter:
#         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum / n
def evaluate_accuracy(data_iter, net):
    """
    计算mini-batch数据上给定的模型的正确率（针对带有dropout的模型进行适配）。
    """
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            # 如果模型是通过torch来定义的（必然是nn.Module），
            # 则torch会根据net当前是处于评估模式（eval）还是训练模式（train）来进行抉择是否dropout
            # 评估模型自然要进入eval模式，但是别忘了评估结束后切换回来
            net.eval()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()
        else:
            # 模型时从零开始实现的
            if('is_training' in net.__code__.co_varnames):
                # 若模型使用了dropout，则要将is_training设置为false
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()


class FlattenLayer(torch.nn.Module):
    """
    将输入的各特征平铺成向量。
    """
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)


def general_train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    """
    本方法对大多数模型适用，因此写成通用的形式。
    本函数中同时实现了“从零开始实现”以及“借助torch”实现的训练代码。
    """
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            # 计算梯度并根据MGD更新参数
            l.backward()
            if optimizer is None:
                mgd(params, lr, batch_size)
            else:
                optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    """
    绘制对数y和x之间的函数图像。x取迭代次数，y取loss。
    """
    use_svg_display()
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)      # 绘制(x, log(y))
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)