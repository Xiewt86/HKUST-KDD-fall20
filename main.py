from models import *
from dataset.dataset import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_batch(model, batch_size, data_feeder):
    criterion = nn.CrossEntropyLoss()
    outputs = torch.zeros((batch_size, 2), device=device)
    labels = torch.zeros((batch_size,), device=device)
    for i in range(batch_size):
        data_seek1, data_serve1, data_seek2, data_serve2, data_profile1, data_profile2, label = data_feeder.get_pair()
        output = model(data_seek1, data_serve1, data_seek2, data_serve2, data_profile1, data_profile2)
        outputs[i] = output
        labels[i] = label
    return criterion(outputs, labels.long())


def train(model, itr_total=1000, batch_size=128, lr=1e-5, weight_decay=0):
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
    model = SiameseNet(4).to(device)
    model = model.float()
    train(model)
