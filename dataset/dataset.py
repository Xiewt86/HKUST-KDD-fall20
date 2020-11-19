from utils import *
import torch
import random


class MyDataset:
    def __init__(self):
        self.raw_trajs = load_data('./dataset/', 'trajs_with_speed500.pkl')
        self.profiles = load_data('./dataset/', 'profile_features500.pkl')
        self.plates = load_data('./dataset/', 'plates.pkl')
        self.plates.remove('d1329')
        self.plates = self.plates[:500]

    def get_item(self, plate, day_id, work_type, task_idx, speed=True):
        if speed:
            cols = [0, 1, 2, 3]
        else:
            cols = [0, 1, 2]
        data_np = np.array(self.raw_trajs[plate][day_id][work_type][task_idx])[:, cols]
        data_tensor = torch.from_numpy(data_np).float()
        data_tensor = data_tensor.reshape((1, data_tensor.shape[0], data_tensor.shape[1]))
        return data_tensor

    def get_profile(self, plate, day_id):
        data_np = np.array(self.profiles[plate][day_id])
        data_tensor = torch.from_numpy(data_np).float()
        data_tensor = data_tensor.reshape((1, data_tensor.shape[0]))
        return data_tensor


class DataFeeder:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_pair(self):
        day_id1 = random.randint(0, 9)
        day_id2 = random.randint(0, 9)
        plate1 = random.sample(self.dataset.plates, 1)[0]
        if random.random() <= 0.5:
            plate2 = plate1
            label = torch.zeros(1)
        else:
            label = torch.ones(1)
            plate2 = random.sample(self.dataset.plates, 1)[0]
            while plate1 == plate2:
                plate2 = random.sample(self.dataset.plates, 1)[0]
        task_seek1 = random.randint(0, len(self.dataset.raw_trajs[plate1][day_id1]['seek']) - 1)
        task_seek2 = random.randint(0, len(self.dataset.raw_trajs[plate2][day_id2]['seek']) - 1)
        data_seek1 = self.dataset.get_item(plate1, day_id1, 'seek', task_seek1)
        data_seek2 = self.dataset.get_item(plate2, day_id2, 'seek', task_seek2)

        task_serve1 = random.randint(0, len(self.dataset.raw_trajs[plate1][day_id1]['serve']) - 1)
        task_serve2 = random.randint(0, len(self.dataset.raw_trajs[plate2][day_id2]['serve']) - 1)
        data_serve1 = self.dataset.get_item(plate1, day_id1, 'serve', task_serve1)
        data_serve2 = self.dataset.get_item(plate2, day_id2, 'serve', task_serve2)

        data_profile1 = self.dataset.get_profile(plate1, day_id1)
        data_profile2 = self.dataset.get_profile(plate2, day_id2)

        return data_seek1, data_serve1, data_seek2, data_serve2, data_profile1, data_profile2, label
