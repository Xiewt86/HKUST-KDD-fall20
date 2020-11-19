from models import *
from dataset.dataset import *
import torch
import getopt
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    return criterion(outputs, labels)


def train(model, itr_total=1000, batch_size=256, lr=1e-4, weight_decay=0.0):
    dataset = MyDataset()
    data_feeder = DataFeeder(dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for itr in range(itr_total):
        loss = loss_batch(model, batch_size, data_feeder)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Iteration: {}\tLoss: {}'.format(itr, loss.detach().cpu().numpy()))


if __name__ == "__main__":
    b, its, lr, wd = parse_args(sys.argv[1:])
    model = SiameseNet(4).to(device)
    model = model.float()
    train(model, itr_total=its, batch_size=b, lr=lr, weight_decay=wd)
