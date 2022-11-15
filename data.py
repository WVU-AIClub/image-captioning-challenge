from torch.utils.data import Dataset
import torchvision.transforms as T

class BloomData(Dataset):
    def __init__(self, img_size, lang):
        preprocess = T.Compose([
            T.CenterCrop(),
            T.Resize(img_size)
        ])
        
        # Load images here
        
    def __len__(self):
        return

    def __getitem__(self, idx):

        return

    def load_images(self):
        return

    def load_label_map(self):
        return