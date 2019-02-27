import random
import mxnet as mx
from mxnet import gluon, nd, autograd
import numpy as np

batch_size = 128
epochs = 5
ctx = mx.gpu() if len(mx.test_utils.list_gpus()) > 0 else mx.cpu()
lr = 0.01


train_dataset = gluon.data.vision.MNIST(train=True)
test_dataset = gluon.data.vision.MNIST(train=False)

def transform(x,y):
    x = x.transpose((2,0,1)).astype('float32')/255.
    y1 = y
    y2 = y % 2 #odd or even
    return x, np.float32(y1), np.float32(y2)


train_dataset_t = train_dataset.transform(transform)
test_dataset_t = test_dataset.transform(transform)

train_data = gluon.data.DataLoader(train_dataset_t, shuffle=True, last_batch='rollover', batch_size=batch_size, num_workers=5)
test_data = gluon.data.DataLoader(test_dataset_t, shuffle=False, last_batch='rollover', batch_size=batch_size, num_workers=5)


class MultiTaskNetwork(gluon.HybridBlock):

    def __init__(self):
        super(MultiTaskNetwork, self).__init__()

        self.shared = gluon.nn.HybridSequential()
        with self.shared.name_scope():
            self.shared.add(
                gluon.nn.Dense(128, activation='relu'),
                gluon.nn.Dense(64, activation='relu'),
                gluon.nn.Dense(10, activation='relu')
            )
        self.output1 = gluon.nn.Dense(10)  # Digits recognition
        self.output2 = gluon.nn.Dense(1)  # odd or even

    def hybrid_forward(self, F, x):
        y = self.shared(x)
        output1 = self.output1(y)
        output2 = self.output2(y)
        return output1, output2

loss_digits = gluon.loss.SoftmaxCELoss()
loss_odd_even = gluon.loss.SigmoidBCELoss()
acc_digits = mx.metric.Accuracy(name='digits')
acc_odd_even = mx.metric.Accuracy(name='odd_even')

net = MultiTaskNetwork()


net.initialize(mx.init.Xavier(), ctx=ctx)
#net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':lr})


def evaluate_accuracy(net, data_iterator):
    acc_digits = mx.metric.Accuracy(name='digits')
    acc_odd_even = mx.metric.Accuracy(name='odd_even')

    for i, (data, label_digit, label_odd_even) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label_digit = label_digit.as_in_context(ctx)
        label_odd_even = label_odd_even.as_in_context(ctx).reshape(-1, 1)

        output_digit, output_odd_even = net(data)

        acc_digits.update(label_digit, output_digit.softmax())
        acc_odd_even.update(label_odd_even, output_odd_even.sigmoid() > 0.5)
    return acc_digits.get(), acc_odd_even.get()


loss_factors = [0.5, 0.5] # Combine losses factor


def multi_task_loss(outputs, labels, loss_factors):
    l_digits = loss_digits(outputs[0], labels[0])
    l_odd_even = loss_odd_even(outputs[1], labels[1])
    return l_digits * loss_factors[0] + l_odd_even * loss_factors[1]

def update_metric(outputs, labels):
    acc_digits.update(labels[0], outputs[0].softmax())
    acc_odd_even.update(labels[1], outputs[1] > 0.5)

def fit(net, train_data, metrics, epochs):

    for e in range(epochs):

        for metric in metrics:
            metric.reset()

        for i, batch_data in enumerate(train_data):
            data = batch_data[0]
            labels = batch_data[1:]
            data = data.as_in_context(ctx)
            for label in labels:
                label.as_in_context(ctx)


            with autograd.record():
                outputs = net(data)
                loss = multi_task_loss(outputs, labels, loss_factors)

            loss.backward()
            trainer.step(data.shape[0])
            update_metric(outputs, labels)

        print(acc_digits.get()[1])
        print(acc_odd_even.get()[1])

        print("Epoch [{}], Acc Digits   {:.4f}".format(
            e, acc_digits.get()[1]))
        print("Epoch [{}], Acc Odd/Even {:.4f}".format(
            e, acc_odd_even.get()[1]))
        #rint("Epoch [{}], Testing Accuracies {}".format(e, evaluate_accuracy(net, test_data)))

fit(net, train_data, metrics=[acc_digits, acc_odd_even], epochs=10 )
