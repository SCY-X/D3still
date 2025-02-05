import glob
import re
import os.path as osp
from .bases import BaseImageDataset


class CUB200(BaseImageDataset):
    """
    The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is a popular benchmark dataset for fine-grained image 
    classification and recognition tasks. It focuses on bird species and includes detailed annotations, making 
    it suitable for tasks such as object detection, segmentation, and fine-grained classification.

    Dataset details:
    - Total images: 11,788.
    - Number of classes: 200 (bird species).
    - Training set: 5,994 images.
    - Test set: 5,794 images.
    - Each image is annotated with:
        - Class label.
        - Bounding box.
        - Part locations (e.g., beak, wing, tail).
        - Attributes (e.g., color, shape).

    This dataset is widely used to study fine-grained recognition due to its challenging inter-class and intra-class 
    variability.

    Dataset download link:
    - Homepage: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    - Direct link: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
    """

    dataset_dir = 'CUB_200_2011'

    def __init__(self, root='../', verbose=True, **kwargs):
        super(CUB200, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> SOP loaded")
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
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path.split("/")[-1]).groups())
            #if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path.split("/")[-1]).groups())
            assert 0 <= pid <= 200  # pid == 0 means background
            assert 0 <= camid <= 11787
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

