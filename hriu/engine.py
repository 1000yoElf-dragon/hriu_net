import numpy as np
from matplotlib import pyplot as plt
from IPython.display import ProgressBar
from torch import cuda, no_grad

from .utils import Absolute


class Metric:
    def __init__(self, func, name=None):
        if isinstance(func, Metric):
            self.func = func.func
            self.name = func.name
        else:
            self.func = func
            self.name = name or func.__name__
        self.value = 0.
        self.total = 0.

    def update(self, ypred, ytrue):
        metric = self.func(ypred, ytrue)
        self.value += metric.sum().item()
        self.total += metric.numel()
        return metric

    def get(self):
        return self.value / (self.total or 1)

    def reset(self):
        self.value = 0.
        self.total = 0.


def accuracy(ypred, ytrue):
    return ypred.argmax(dim=-1) == ytrue


class Accuracy(Metric):
    def __init__(self):
        super().__init__(accuracy)
        self.reset()


class History(list):
    """
    Records history as list of runs. For every run appends a new dict with structure:
    {
        modename1: {
                       'epoch': list of epochs
                       'metrics': {
                                      metricname1: list of metric 1 values on epoch
                                      metricname2: ...
                                  }
                   }
        modename2: ...
    }
    where modename is usually 'train' or 'test'
    """

    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics
        self.epoch = 0

    def new_run(self):
        self.epoch = 0
        self.append(dict())

    def update(self, mode, epoch=None):
        if mode in self[-1]:
            data = self[-1][mode]
        else:
            dict_ = {
                'epoch': [],
                'metrics': dict((metric.name, []) for metric in self.metrics)
            }
            data = self[-1][mode] = dict_

        self.epoch = epoch or self.epoch
        data['epoch'].append(self.epoch)
        for metric in self.metrics:
            data['metrics'][metric.name].append(metric.get())
        self.epoch += 1

    def plot(self, run=None, scale='linear'):

        if isinstance(run, str) and run.lower() == 'all':
            runs = self
        else:
            run = run or -1
            run = run if run >= 0 else (len(self) + run)
            runs = [self[run]]

        log_scale = set()
        if isinstance(scale, dict):
            for k, v in scale.items():
                if v.lower() == 'log':
                    log_scale.add(k)
        elif scale.lower() == 'log':
            log_scale = Absolute()

        plots = dict()

        add_epoch = 0
        for r in runs:
            max_epoch = 0
            for mode, a in r.items():
                epoch = [x + add_epoch for x in a['epoch']]
                max_epoch = max(max_epoch, max(epoch))
                for name, metric in a['metrics'].items():
                    if name not in plots:
                        plots[name] = {mode: {'epoch': epoch, 'metric': metric}}
                    else:
                        if mode not in plots[name]:
                            plots[name][mode] = {'epoch': epoch, 'metric': metric}
                        else:
                            plots[name][mode]['epoch'] += epoch
                            plots[name][mode]['metric'] += metric
            add_epoch = max_epoch + 1

        n = len(plots)
        rows = (n - 1) // 2 + 1
        fig, axes = plt.subplots(rows, 1 + (n > 1), squeeze=False)
        fig.set_size_inches(16, 7 * rows)
        axes = axes.flatten()

        for ax, (name, a) in zip(axes, plots.items()):
            legend = list()
            for mode, data in a.items():
                if name in log_scale:
                    ax.semilogy(data['epoch'], data['metric'])
                else:
                    ax.plot(data['epoch'], data['metric'])
                legend.append(mode)

            ax.grid(True)
            ax.legend(legend)
            ax.set_ylabel(name)
            ax.set_xlabel('Epoch')
            ax.set_title(f'run {run}')
        return fig


class Engine:
    def __init__(self, model, train_loader, test_loader, optimizer, loss_fn, metrics=None, device=None):
        self.device = device or ('cuda' if cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss = Metric(loss_fn, 'loss')
        if isinstance(metrics, (list, tuple)):
            self.metrics = [Metric(metric) for metric in metrics]
        elif isinstance(metrics, dict):
            self.metrics = [Metric(metric, key) for key, metric in metrics.items()]
        elif callable(metrics):
            self.metrics = [Metric(metrics)]
        else:
            self.metrics = []
        self.history = History([self.loss] + self.metrics)

    # Обучение на одной эпохе
    def train_epoch(self):
        # Переключаем в режим обучения
        self.model.train()

        # Создаем и выводим индикатор прогресса
        length = len(self.train_loader.dataset)
        progress_bar = ProgressBar(length)
        progress_bar.display()

        self.loss.reset()
        for metric in self.metrics:
            metric.reset()

        # Цикл по всем пакетам
        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)

            loss = self.loss.update(pred, y)

            # Пересчитываем веса
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            progress_bar.progress += y.shape[0]
            string = 'Train {}/{}: loss = {:.4G}'.format(progress_bar.progress, length, self.loss.get())

            for metric in self.metrics:
                metric.update(pred, y)
                string += ', ' + metric.name + ' = {:.4G}'.format(metric.get())

            print('\r', string, end='')
        print()
        if len(self.metrics) > 0:
            return (self.loss.get(),) + tuple(metric.get for metric in self.metrics)
        else:
            return self.loss.get()

    def test_epoch(self):
        # Переключаем в режм расчета
        self.model.eval()

        length = len(self.test_loader.dataset)
        done = 0

        # Зануляем loss и метрику
        self.loss.reset()
        for metric in self.metrics:
            metric.reset()

        with no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)

                self.loss.update(pred, y)
                done += y.shape[0]
                string = 'Test {}/{}: loss = {:.4G}'.format(done, length, self.loss.get())
                for metric in self.metrics:
                    metric.update(pred, y)
                    string += ', ' + metric.name + ' = {:.4G}'.format(metric.get())
                print('\r', string, end='')
        print()

        if len(self.metrics) > 0:
            return (self.loss.get(),) + tuple(metric.get for metric in self.metrics)
        else:
            return self.loss.get()

    def fit(self, epochs, scale=None):
        if scale is None:
            scale = {'loss': 'log'}
        self.history.new_run()
        for epoch in range(epochs):
            # Обучаем и запоминаем ошибку
            print('Epoch: ', epoch)
            self.train_epoch()
            self.history.update(mode='train')

            # Проверяем
            self.test_epoch()
            self.history.update(mode='test')

        self.history.plot(scale=scale)
        plt.show()
