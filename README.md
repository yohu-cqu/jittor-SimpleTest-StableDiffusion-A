# DreamBooth-Lora

本项目参考自 [JDiffusion 的 DreamBooth-Lora](https://github.com/JittorRepos/JDiffusion/tree/master/examples/dreambooth) 。

## 环境安装

按照 [JDiffusion 的环境安装一节](https://github.com/JittorRepos/JDiffusion/blob/master/examples/dreambooth/README.md) 安装必要的依赖。

接着设置运行脚本的权限：
```
chmod u+x ./dreambooth/*.sh
```

## 训练

1. 首先从比赛云盘下载对应的数据集，推荐将数据集目录 `A` 下载到 `./A` 下
2. 将 `train_all.sh` 中的 `HF_HOME` 设置为本地模型路径， `root` 设置为项目所在目录， `BASE_INSTANCE_DIR` 设置为数据集对应的目录，`GPU_COUNT` 设置为对应可用的显卡数量，`MAX_NUM` 设置为数据集中的风格个数；
3. 然后进入目标文件夹下： `cd ./dreambooth/`， 运行 `bash train_all.sh` 即可训练。保存的模型会存放至 `./dreambooth/results/prompt_v1_color_test1/style_[训练epoch数]epoch` 目录下，例如：`./dreambooth/results/prompt_v1_color_test1/style_300epoch`。

## 推理

1. 将 `test_all.sh` 中的 `HF_HOME` 设置为本地模型路径，将 `run_all.py` 中的 `root` 设置为项目所在目录， `dataset_root` 修改为数据集对应的目录，将 `max_num` 修改为数据集中的风格个数；
2. 进入目标文件夹下： `cd ./dreambooth/`，运行 `bash test_all.sh` 进行推理。模型生成的图片会输出到 `./dreambooth/results/prompt_v1_color_test1/outputs_[保存点训练epoch数]ckpt_[推理轮数]steps_[种子值]seed` 文件夹下，例如：`./dreambooth/results/prompt_v1_color_test1/outputs_300ckpt_500steps_76587seed`。


## 参考文献

```
@inproceedings{ruiz2023dreambooth,
  title={Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
