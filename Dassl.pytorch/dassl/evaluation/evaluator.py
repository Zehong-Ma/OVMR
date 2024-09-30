import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt, topk=1):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        if topk==1:
            pred = mo.max(1)[1]
            matches = pred.eq(gt).float()
        else:
            pred = mo.topk(k=topk, dim=-1)[1]
            matches = (pred==(gt.unsqueeze(1).repeat(1,topk))).float().sum(dim=-1)
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())

        if topk>1:
            self._y_pred.extend(pred[:,0].data.cpu().numpy().tolist())
        else:
            self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)
    

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        # import pdb
        # pdb.set_trace()
        # acc_per_class
        self._y_pred[1==self._y_true]
        unique_labels = np.unique(np.array(self._y_true))
        my_dict = {}
        for label_ in unique_labels:
            pred_per_class = np.array(self._y_pred)[label_==np.array(self._y_true)]
            acc_per_class = 100*(pred_per_class==label_).sum()/len(pred_per_class)
            # print("class %d, acc: %.3f"%(label_, acc_per_class))
            my_dict[str(label_)] = acc_per_class
        import csv
        with open(self.cfg.OUTPUT_DIR+"/acc_per_class.csv", mode='w', newline='') as csv_file:
       
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['Label', 'Acc'])

            for key, value in sorted(my_dict.items()):
                csv_writer.writerow([key, value])

        f1_per_class = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average=None,
            labels=np.unique(self._y_true)
        )
        # import pdb
        # pdb.set_trace()
        with open(self.cfg.OUTPUT_DIR+"/f1_per_class.csv", mode='w', newline='') as csv_file:
       
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['Label', 'F1'])

            for item_id, value in enumerate(f1_per_class):
                csv_writer.writerow([item_id, value])


        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
