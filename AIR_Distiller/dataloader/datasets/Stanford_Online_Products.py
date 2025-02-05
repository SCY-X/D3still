import glob
import re
import os.path as osp
from .bases import BaseImageDataset


class SOP(BaseImageDataset):
    """
    The Stanford Online Products (SOP) dataset is a large-scale dataset commonly used for metric learning tasks.
    It contains 22,634 product categories, with each category represented by multiple images collected from an 
    online e-commerce platform. The dataset is split into a training set and a test set:
    - Training set: 11,318 categories, 59,551 images.
    - Test set: 11,316 categories, 60,502 images.

    The primary task of this dataset is to learn the similarity relationships between images for evaluating 
    performance in tasks such as image retrieval, clustering, and classification.

    Dataset download link:
    - Homepage: https://cvgl.stanford.edu/projects/lifted_struct/
    - Direct link: ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
    """
    
    dataset_dir = 'Stanford_Online_Products'

    def __init__(self, root='../', verbose=True, **kwargs):
        super(SOP, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Stanford Online Products loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.JPG'))
        pattern = re.compile(r'([-\d]+)_(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid = pid - 1
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            pid = pid - 1
            camid = camid - 1
            assert 0 <= pid <= 22633  # pid == 0 means background
            assert 0 <= camid <= 120052
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

