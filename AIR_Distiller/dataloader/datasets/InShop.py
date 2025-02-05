import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class InShop(BaseImageDataset):
    """
    The In-Shop Clothes Retrieval (InShop) dataset is a benchmark dataset designed for evaluating image retrieval
    and person re-identification tasks in an e-commerce context. It contains images of clothing items collected 
    from an online shopping platform, split into two main tasks: consumer-to-shop retrieval and category-level 
    image retrieval.

    Dataset details:
    - Total images: 54,642.
    - Training set: 3,997 categories, 25,882 images.
    - Query set: 3,985 categories, 14,218 images.
    - Gallery set: 3,985 categories, 12,612 images.

    In the consumer-to-shop retrieval task, the query set contains images of clothing items taken by customers,
    while the gallery set contains professionally taken images of the same items.

    Dataset download link:
    - Homepage: https://github.com/xyguo/fashion-iq
    - Direct link: https://drive.google.com/file/d/1gGAW47kvC9QEk8E4UpEpA6Osh0GGUqby/view
    """
    
    dataset_dir = 'InShop'

    def __init__(self, root='../', verbose=True, **kwargs):
        super(InShop, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, mode='train')
        query = self._process_dir(self.query_dir, relabel=False, mode='query')
        gallery = self._process_dir(self.gallery_dir, relabel=False, mode='gallery')

        if verbose:
            print("=> InShop loaded")
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

    def _process_dir(self, dir_path, relabel=False, mode='train'):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_')

        if relabel:
            pid_container = set()
            for img_path in img_paths:
                pid = map(int, pattern.search(img_path).groups())
                pid = list(pid)[0] - 1
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid = map(int, pattern.search(img_path).groups())
            pid = list(pid)[0] - 1
            if mode == 'train' or mode == 'query':
                camid = 0
            else:
                camid = 1
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

