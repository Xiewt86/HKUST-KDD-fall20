import getopt
import logging
import sys
import time

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
    its = 10000
    lr = 1e-4
    weight_decay = 1e-4
    log_every = 100
    n_plates = 500
    n_days = 10
    log_name = 'torch.log'
    hlp_msg = 'python main.py -b <batch size> -i <iteration> -l <learning rate> -w <weight decay> -e <log every> -n <log name> -p <num plates> -d <num days>'
    try:
        opts, args = getopt.getopt(argv, "hb:i:l:w:e:n:p:d:")
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
        elif opt == "-e":
            log_every = int(arg)
        elif opt == "-n":
            log_name = str(arg)
        elif opt == "-p":
            n_plates = int(arg)
        elif opt == "-d":
            n_days = int(arg)
    return batch_size, its, lr, weight_decay, log_every, log_name, n_plates, n_days


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


def train(model, itr_total=10000, batch_size=256, lr=1e-4, weight_decay=0.0, log_every=100, n_plates=500, n_days=10):
    start_time = time.time()
    dataset_train = MyDataset(num_plates=n_plates)
    data_feeder_train = DataFeeder(dataset_train, np.arange(0, n_days))
    dataset_test = MyDataset(train=False)
    data_feeder_test = DataFeeder(dataset_test, np.arange(n_days, 10))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for itr in range(itr_total):
        loss_train, outputs_train, labels_train = loss_batch(model, batch_size, data_feeder_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        train_acc = acc(outputs_train.detach().cpu().numpy(), labels_train.detach().cpu().numpy())
        print('Iteration: {}\tLoss: {:1.4f}\tTraining acc: {:1.4f}'.format(itr, loss_train.detach().cpu().numpy(),
                                                                           train_acc))

        if (itr % log_every == 0) & (itr != 0):
            loss_test, outputs_test, labels_test = loss_batch(model, 1000, data_feeder_test)
            test_acc = acc(outputs_test.detach().cpu().numpy(), labels_test.detach().cpu().numpy())
            print('----------------------- Validation:\tLoss: {:1.4f}\tValidation acc: {:1.4f}'.format(
                loss_test.detach().cpu().numpy(),
                acc(outputs_test.detach().cpu().numpy(),
                    labels_test.detach().cpu().numpy())))
            logging.info('******iteration: ' + str(itr) + '; loss: ' + str(loss_test) + '; train acc: ' + str(
                train_acc) + '; test acc: ' + str(test_acc))
    logging.info('total running time: ' + str(time.time() - start_time))


if __name__ == "__main__":
    b, its, lr, wd, log, log_name, n_plates, n_days = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=log_name,
                        filemode='a')
    model = SiameseNet(4).to(device)
    model = model.float()
    train(model, itr_total=its, batch_size=b, lr=lr, weight_decay=wd, log_every=log, n_plates=n_plates, n_days=n_days)
