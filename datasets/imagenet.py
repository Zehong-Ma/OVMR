import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            # preprocessed_eval_seen = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}-eval-seen.pkl")
            # preprocessed_eval_unseen = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}-eval-unseen.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                # eval_seen = self.generate_fewshot_dataset_eval(train, num_shots=num_shots, is_seen=True, seed=seed)
                # eval_unseen = self.generate_fewshot_dataset_eval(train, num_shots=num_shots, is_seen=False, seed=seed) # select from training set
                
                data = {"train": train}
                # print(train)
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            # file_paths = [for item in train]
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)
       
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
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir()) # scan the dataset dir to check which class the folder should belong to.
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                print(impath)
        return items
