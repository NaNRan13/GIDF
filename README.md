# Learning Guided Implicit Depth Function with Scale-aware Feature Fusion
## Trained Models
```
https://drive.google.com/drive/folders/1ed_PN-xnZI0ZveWp37OuIalcwf7g32AF?usp=drive_link
```

## Dateset
```
We train on NYUv2 dataset, then test on NYUv2, Middlebury, Lu and RGBDD dataset
```

## Train
```
GIDF-S:
CUDA_VISIBLE_DEVICES='0' python main.py --name gidf_s2_b3 --depth 2 --num_bif 3 --checkpoint scratch
GIDF-S*:
CUDA_VISIBLE_DEVICES='0' python main.py --name gidf_s2_b3_2stage --depth 2 --num_bif 3 --checkpoint scratch --AR_epoch 200 --epoch 400
GIDF-L*:
CUDA_VISIBLE_DEVICES='0,1' python main_para.py --name gidf_s4_b4_2stage  --para --para_block 2  --depth 4 --num_bif 4  --checkpoint scratch --AR_epoch 200 --epoch 400
```

## Test
```
GIDF-S:
CUDA_VISIBLE_DEVICES='0' python main.py --name gidf_s2_b3 --test --scale 4 --scale_max 4 --save --batched_eval --checkpoint best   --dataset Middlebury --data_root ./data/Middlebury/
GIDF-S*:
CUDA_VISIBLE_DEVICES='0' python main.py --name gidf_s2_b3_2stage --test --scale 4 --scale_max 4 --save --batched_eval --checkpoint best  --dataset Middlebury --data_root ./data/Middlebury/
GIDF-L*:
CUDA_VISIBLE_DEVICES='0,1' python main_para.py --name gidf_s4_b4_2stage --test --scale 4 --scale_max 4 --save --batched_eval --checkpoint best  --para --para_block 2  --depth 4 --num_bif 4 --dataset Middlebury --data_root ./data/Middlebury/
```

## Citation
```
@article{GIDF,
  author={Zuo, Yifan and Hu, Yuqi and Xu, Yaping and Wang, Zhi and Fang, Yuming and Yan, Jiebin and Jiang, Wenhui and Peng, Yuxin and Huang, Yan},
  journal={IEEE Transactions on Image Processing}, 
  title={Learning Guided Implicit Depth Function With Scale-Aware Feature Fusion}, 
  year={2025},
  volume={34},
  pages={3309-3322}
}
```
