# Open-Vocabulary Recognition with Multi-Modal References

## Installation

See [installation instructions](docs/INSTALL.md).


## Benchmark evaluation and training

Please first [prepare datasets](datasets/README.md), then check our [MODEL ZOO](docs/MODEL_ZOO.md) to reproduce results in our paper.


## License

See [Detic](https://github.com/facebookresearch/Detic). Our code is based on this repository.

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{Kaul2023,
      title={Multi-Modal Classifiers for Open-Vocabulary Object Detection},
      author={Kaul, Prannay and Xie, Weidi and Zisserman, Andrew},
      booktitle={ICML},
      year={2023}
    }

python demo.py --config-file configs/mm_classifier_swin_b_box_visualization.yaml --input datasets/coco/val2017/*.jpg --output visualization_img_output_mask_novel --vocabulary lvis --confidence-threshold 0.5 --opts MODEL.WEIGHTS checkpoints/model_final_swinb.pth

python demo.py --config-file configs/mm_classifier_swin_b_box_visualization.yaml --input visualization_img_input/1.mp4 --output visualize_videos --vocabulary lvis --confidence-threshold 0.5 --opts MODEL.WEIGHTS checkpoints/model_final_swinb.pth


114770.jpg

8211
16228

16958