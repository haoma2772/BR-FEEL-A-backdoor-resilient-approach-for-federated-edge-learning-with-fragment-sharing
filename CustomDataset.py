
from torch.utils.data import Dataset

class CustomPoisonedDataset(Dataset):
    # just use the img and label
    def __init__(self, img_set, poison_indices, label_set, cover_indices=None):
        self.img_set = img_set
        self.poison_indices = poison_indices
        self.cover_indices = cover_indices if cover_indices is not None else []
        self.label_set = label_set

    def __len__(self):
        return len(self.img_set)

    def __getitem__(self, idx):
        idx = int(idx)
        img = self.img_set[idx]  # GET THE INCORRESPONDING IMAGE
        label = self.label_set[idx]  # GET THE INCORRESPDOING LABEL
        return img, label