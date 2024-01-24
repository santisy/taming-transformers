from taming.data.base_zip import ImageFolderDataset

class LSUNChurchTrain(ImageFolderDataset):
    def __init__(self, data_path='datasets/lsunchurch_for_stylegan.zip', **kwargs):
        super().__init__(path=data_path)

class LSUNChurchVal(ImageFolderDataset):
    def __init__(self, data_path='datasets/lsunchurch_for_stylegan_val.zip', **kwargs):
        super().__init__(path=data_path)
