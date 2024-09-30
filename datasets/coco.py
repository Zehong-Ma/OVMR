import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets



@DATASET_REGISTRY.register()
class COCO(DatasetBase):

    dataset_dir = "coco_exemplars_mmovod_K30_final"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        # if cfg.DATASET.NUM_SHOTS == 40:
        #     self.preprocessed = os.path.join(self.dataset_dir, "preprocessed_ovd.pkl")
        
        # if os.path.exists(self.preprocessed):
        #     with open(self.preprocessed, "rb") as f:
        #         preprocessed = pickle.load(f)
        #         train = preprocessed["train"]
        #         test = preprocessed["test"]
        # else:
            # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = self.read_classnames(COCO_CLASSES)
       
        train = self.read_data(classnames, "train")
        # Follow standard practice to perform evaluation on the val set
        # Also used as the val set (so evaluate the last-step model)
        test = self.read_data(classnames, "val")

        preprocessed = {"train": train, "test": test}
        # with open(self.preprocessed, "wb") as f:
        #     pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            # preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            # preprocessed_eval_seen = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}-eval-seen.pkl")
            # preprocessed_eval_unseen = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}-eval-unseen.pkl")
            # if os.path.exists(preprocessed):
            #     print(f"Loading preprocessed few-shot data from {preprocessed}")
            #     with open(preprocessed, "rb") as file:
            #         data = pickle.load(file)
            #         train = data["train"]
            # else:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                # eval_seen = self.generate_fewshot_dataset_eval(train, num_shots=num_shots, is_seen=True, seed=seed)
                # eval_unseen = self.generate_fewshot_dataset_eval(train, num_shots=num_shots, is_seen=False, seed=seed) # select from training set
                
            data = {"train": train}
                
                # print(f"Saving preprocessed few-shot data to {preprocessed}")
                # with open(preprocessed, "wb") as file:
                #     pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        
        train, test = OxfordPets.subsample_COCO_classes(train, test, subsample=subsample)
        if subsample=="all":
            super().__init__(train_x=test, val=test, test=train, eval_set=test)
        else:
            super().__init__(train_x=train, val=test, test=test, eval_set=train)

    def generate_fewshot_dataset_eval(
        self, *data_sources, num_shots=-1, repeat=False, is_seen=True, seed=1, exist_few_shot_train=None
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        import random
        random.seed(seed)
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset for eval")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            if exist_few_shot_train is not None:
                tracker_exist = self.split_dataset_by_label(exist_few_shot_train)
            dataset = []

            for label, items in tracker.items():
                if is_seen is False:
                    if len(items) >= num_shots:
                        sampled_items = random.sample(items, num_shots)
                    else:
                        if repeat:
                            sampled_items = random.choices(items, k=num_shots)
                        else:
                            sampled_items = items
                else:
                    if len(items) >= 2*num_shots:
                        items = random.shuffle(items)
                        i = 0
                        sampled_items = []
                        exist_file_paths = [ite.impath for ite in tracker_exist[label]]
                        for item in items:
                            if item.impath in exist_file_paths:
                                continue
                            else:
                                sampled_items.append(item)
                                i+=1
                                if i==num_shots:
                                    break
                        assert len(sampled_items)==num_shots
                    else:
                        print(len(items))
                        print("there are classes less than 2*num_shot!!!")
                        raise Exception("there are classes less than 2*num_shot!!!")
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    @staticmethod
    def read_classnames(COCO_classes):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        for ind, class_ in enumerate(COCO_classes):
            folder = str(ind)
            classnames[folder] = class_
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir()) # scan the dataset dir to check which class the folder should belong to.
        items = []

        for label, folder in enumerate(folders):
            label = int(folder)
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items


COCO_CLASSES_DICT = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

COCO_CLASSES =[item_[1] for item_ in sorted(COCO_CLASSES_DICT.items(), key=lambda x: int(x[0]))]
# print(len(COCO_CLASSES))