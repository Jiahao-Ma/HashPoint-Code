<p align="center">

  <h1 align="center"> HashPoint: Accelerated Point Searching and Sampling for Neural Rendering </h1>
  
  <h3 align="center"> CVPR 2024 (Highlight)</h3>

</p>

<h3 align="center"><a href="https://arxiv.org/abs/2404.14044v1">Paper</a> | <a href="https://www.youtube.com/watch?v=3tUB-ewh3LY">Video</a> | <a href="https://jiahao-ma.github.io/hashpoint/">Project Page</a></h3>
<div align="center"></div>
<p align="center">
  <a href="https://jiahao-ma.github.io/hashpoint/hashpoint/video2.mp4">
    <a href="">
    <img src="./media/video2.gif" alt="teaser" width="100%">
  </a>
  </a>
</p>

<p align="left">
The project provides official implementation of <a href="https://arxiv.org/abs/2404.14044v1">HashPoint</a> in CVPR'24. The HashPoint method significantly improves the efficiency of point searching and sampling by combining rasterization with ray-tracing techniques. Unlike the multi-surface sampling approach of Point-NeRF, HashPoint concentrates on the primary surface, thereby acclerating the rendering process. The video above demonstrates this streamlined approach in action.
</p>

## TODOs
- [x] The source code of hashpoint (Searching and Sampling)
- [ ] Integration with Point-NeRF (Coming soon)
- [ ] The detailed instructions (Coming soon)

### Installation

1. Create a new conda environment named `mvchm` to run this project:
```
conda create -n hashpoint python=3.9.7
conda activate hashpoint
```

2. Make sure your system meet the CUDA requirements and install some core packages:
```
pip install easydict torch==1.12.1+cu113 torchvision==0.13.1+cu113 tqdm scipy opencv-python
```

3. Clone this repository
```
cd Your-Project-Folder
gir clone git@github.com:Jiahao-Ma/HashPoint-Code.git
```

4. Install hashpoint CUDA library 
```
pip install .
```