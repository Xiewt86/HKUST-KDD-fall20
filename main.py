import getopt
import sys

import torch

from dataset.dataset import *
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def acc(outputs, labels):
    total = len(labels)
    cnt = 0.0
    for i in range(len(outputs)):
        if round(outputs[i]) == labels[i]:
            cnt += 1
    return cnt / total


def parse_args(argv):
    batch_size = 128
    its = 1000
    lr = 1e-4
    weight_decay = 1e-4
    hlp_msg = 'python main.py -b <batch size> -i <iteration> -l <learning rate> -w <weight decay>'
    try:
        opts, args = getopt.getopt(argv, "hb:i:l:w:")
    except getopt.GetoptError:
        print(hlp_msg)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(hlp_msg)
            sys.exit()
        elif opt == "-b":
            batch_size = int(arg)
        elif opt == "-i":
            its = int(arg)
        elif opt == "-l":
            lr = float(arg)
        elif opt == "-w":
            weight_decay = float(arg)
    return batch_size, its, lr, weight_decay


def loss_batch(model, batch_size, data_feeder):
    criterion = nn.BCELoss()
    outputs = torch.zeros((batch_size,), device=device)
    labels = torch.zeros((batch_size,), device=device)
    for i in range(batch_size):
        data_seek1, data_serve1, data_seek2, data_serve2, data_profile1, data_profile2, label = data_feeder.get_pair()
        output = model(data_seek1, data_serve1, data_seek2, data_serve2, data_profile1, data_profile2)
        outputs[i] = output
        labels[i] = label
    return criterion(outputs, labels), outputs, labels


def train(model, itr_total=1000, batch_size=256, lr=1e-4, weight_decay=0.0):
    dataset_train = MyDataset()
    data_feeder_train = DataFeeder(dataset_train)
    dataset_test = MyDataset(train=False)
    data_feeder_test = DataFeeder(dataset_test)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for itr in range(itr_total):
        loss, outputs_train, labels_train = loss_batch(model, batch_size, data_feeder_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Iteration: {}\tLoss: {:1.4f}\tTraining acc: {:1.4f}'.format(itr, loss.detach().cpu().numpy(),
                                                                           acc(outputs_train.detach().cpu().numpy(),
                                                                               labels_train.detach().cpu().numpy())))

        if itr % 20 == 0:
            loss, outputs_test, labels_test = loss_batch(model, 500, data_feeder_test)
            print('----------------------- Validation:\tLoss: {:1.4f}\tValidation acc: {:1.4f}'.format(
                loss.detach().cpu().numpy(),
                acc(outputs_test.detach().cpu().numpy(),
                    labels_test.detach().cpu().numpy())))


if __name__ == "__main__":
    b, its, lr, wd = parse_args(sys.argv[1:])
    model = SiameseNet(4).to(device)
    model = model.float()
    train(model, itr_total=its, batch_size=b, lr=lr, weight_decay=wd)
