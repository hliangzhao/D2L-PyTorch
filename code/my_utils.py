from IPython import display
from matplotlib import pyplot as plt


def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    my_utils.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    # 一张img和对应的lbl填充一个fig
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().get_visible(False)
        f.axes.get_yaxis().get_visible(False)
    plt.show()