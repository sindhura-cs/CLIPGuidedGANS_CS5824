# Generating fashion designs using textual descriptions 

PyTorch implementation of CLIP guided StyleGAN for intereactive fashion design tool. 
Completed as a part of Advanced Machine Learning CS5824 course at Virginia Tech (Spring-2023).

## Pre-trained checkpoints
We have trained the 256px model on FashionIQ dataset with 13500 iterations

Download the pre-trained model: [here](https://drive.google.com/drive/folders/1iczBUx23GC2-ZQ27nd_jJhnZOL00aGF2?usp=share_link)

Run the final model on Collab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tvbPwxB1VnHlMlpUsNBqvkGKkfY08Eti?usp=sharing)

## (1) Requirements
 - Anaconda
 - Pytorch
 - CUDA

## (2) Download FashionIQ Dataset
Image Download script is inculded inside FashionIQData
``` 
run_download_image.sh
```
Create lmdb dataset: python file is included inside FashionIQData
+ all the images are resized to 256px
```
python prepare_data.py --out LMDB_PATH --size 256 DATASET_PATH
```
## (3) Train the image generation model using script inside StyleGAN
Model is trained using infer cluster on [ARC](https://arc.vt.edu/) with 32 cores and 1 GPU
```
sbatch train_stylegan.sh
```

## (4) References
```
@InProceedings{Patashnik_2021_ICCV,
    author    = {Patashnik, Or and Wu, Zongze and Shechtman, Eli and Cohen-Or, Daniel and Lischinski, Dani},
    title     = {StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2085-2094}
}
```
```
@inproceedings{Karras2019stylegan2,
  title     = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author    = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. CVPR},
  year      = {2020}
}
```
```
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```
