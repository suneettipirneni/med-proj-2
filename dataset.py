from torch.utils.data import Dataset
from os import path
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

import json

class BRATSDataset(Dataset):

  def __init__(self, root_dir: str) -> None:
    super().__init__()
    
    # Read dataset.json to get files for training and testing
    dataset_json_file = open(path.join(root_dir, "dataset.json"))
    dataset_json = json.load(dataset_json_file)

    self.file_data = dataset_json['training']
    self.len = dataset_json['numTraining']
    self.root_dir = root_dir


  def __len__(self):
    return self.len

  def __getitem__(self, index: int):
    file_meta = self.file_data[index]
    image_path = path.join(self.root_dir, file_meta['image'])
    label_path = path.join(self.root_dir, file_meta['label'])

    # Load image data
    image: Nifti1Image = nib.load(image_path)

    # Load label data
    label: Nifti1Image = nib.load(label_path)

    return image.get_fdata(), label.get_fdata()
 