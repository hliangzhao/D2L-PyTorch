from IPython import display
from matplotlib import pyplot as plt
import torch
import time
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import zipfile


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


def load_data_fashion_mnist(batch_size, resize=None, root='../data'):
    """
    将fashion-MNIST数据集导入变量train_iter和test_iter。和上面的代码相比增加了resize的选项。
    """
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    # transform is the composition of operates: resize first, and then transform it to tensor
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter


# below code is only suitable for .ipynb before 03-7
# def evaluate_accuracy(data_iter, net):
#     """
#     计算mini-batch数据上给定的模型的正确率。
#     """
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum / n

# below code is only suitable for .ipynb before 05-1
# def evaluate_accuracy(data_iter, net):
#     """
#     计算mini-batch数据上给定的模型的正确率（针对带有dropout的模型进行适配）。
#     """
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         if isinstance(net, torch.nn.Module):
#             # 如果模型是通过torch来定义的（必然是nn.Module），
#             # 则torch会根据net当前是处于评估模式（eval）还是训练模式（train）来进行抉择是否dropout
#             # 评估模型自然要进入eval模式，但是别忘了评估结束后切换回来
#             net.eval()
#             acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#             net.train()
#         else:
#             # 模型时从零开始实现的
#             if('is_training' in net.__code__.co_varnames):
#                 # 若模型使用了dropout，则要将is_training设置为false
#                 acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
#             else:
#                 acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum / n


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0., 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n


class FlattenLayer(torch.nn.Module):
    """
    将输入的各特征平铺成向量。
    """
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)


def general_train(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    """
    本方法对大多数模型适用，因此写成通用的形式。
    本函数中同时实现了“从零开始实现”以及“借助torch”实现的训练代码。
    """
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                mgd(params, lr, batch_size)
            else:
                optimizer.step()


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    """
    本函数给出了torch实现的网络的训练代码。
    本函数允许训练在GPU上执行。
    """
    net = net.to(device)
    print('training on', device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0., 0., 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


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


def corr2d(X, K):
    """
    X是输入的特征映射，K是filter
    """
    h, w = K.shape
    # 窄卷积（N - n + 1）
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[0]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


class GlobalAvgPool2d(nn.Module):
    """
    全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现。
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        # 将单个通道上（宽 * 高个元素的平均值计算出来）
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def load_data_jay_lyrics():
    with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 一个example是一个包含了num_steps时间步的样本
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    # epcoh_size是batch的个数
    batch_num = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)      # 打乱每个样本的顺序（单个样本内仍然是按照原本序列的顺序排列的）
    
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(batch_num):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)        # 给定序列中字符的个数
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
    batch_num = (batch_len - 1) // num_steps     # batch的个数
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def one_hot(x, n_class, dtype=torch.float32):
    # input shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)     # 将x作为索引，把1填充到res的dim=1的对应位置上
    return res


def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= theta / norm


def rnn_train_and_predict(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  
            # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach_()

            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            my_utils.mgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', rnn_predict(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


