# Instance Segmentation of Auroral Images for Automatic Computation of Arc Width

By Chuang Niu, Qiuju Yang, Shenhan Ren, Haihong Hu, Desheng Han, Zejun Hu, and Jimin Liang.

## Introduction
A fully automatic width computation of aurora arcs based on Marsk R-CNN is implemented in this project,
and the related paper is submitted to GRSL. More details will be described.

## Installation

This project is based on [Mask R-CNN](https://github.com/facebookresearch/maskrcnn-benchmark),
[PyTorch 1.0](https://pytorch.org/), and Python 3.5.

1. Install the [Mask R-CNN benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
2. Replace the maskrcnn-benchmark with the aurora-maskrcnn in this project.
3. ```bash
   cd ~/aurora-maskrcnn
   python3 setup.py build develop
   ```

## Demo
Run demo:
```bash
   cd ~/aurora-maskrcnn
   python3 ./Aurora/demo.py
```
It will output the following results:

1. Original image:
![alt text](demo_results/N20040114G105203.png "from http://cocodataset.org/#explore?id=345434")


## Data

## Models

## License

maskrcnn-benchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.
