# Instance Segmentation of Auroral Images for Automatic Computation of Arc Width

By Chuang Niu, Qiuju Yang, Shenhan Ren, Haihong Hu, Desheng Han, Zejun Hu, and Jimin Liang.

## Introduction
A fully automatic method for computing aurora arc width based on Marsk R-CNN is implemented in this project,
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

## Data
Data will be available soon.

## Models
Models will be available soon.

## Demo
Run demo:
```bash
   git clone https://github.com/niuchuangnn/aurora-maskrcnn.git
   cd ~/aurora-maskrcnn
   python3 ./Aurora/demo.py
```
It will output the following results:

1. Original image:
<img width="440" src="/demo_results/N20040116G050623.png"/>

2. Detection results of one-stage inference process:
<img width="440" src="/demo_results/N20040116G050623_one_stage.png"/>

3. Detection results of rotated image:
<img width="440" src="/demo_results/N20040116G050623_two_stage.png"/>

4. Detection results of two-stage inference process:
<img width="440" src="/demo_results/N20040116G050623_two_stage_rotation.png"/>

5. Predicted normal of aurora arcs:
<img width="440" src="/demo_results/N20040116G050623_normal.png"/>

6. Intensity vs. zenith-angle curve:
<img width="440" src="/demo_results/N20040116G050623_intensity.png"/>

## License

maskrcnn-benchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.
