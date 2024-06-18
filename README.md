## 源码地址

```bash
https://github.com/AILab-CVC/YOLO-World
```

## 开始

### 1. 克隆项目
```bash
git clone --recursive https://github.com/2829788992/CV.git
```

### 2. 用anaconda建一个虚拟环境

```bash
conda create --n yolow python=3.8 -y
conda activate yolow
```

### 3. 安装pytorch 

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3
```
### 4. 安装mmengine等OpenMMLab中的库，为了方便安装，可以从pip转为mim进行安装

```bash
pip install -U openmim
mim install mmcv==2.0.1
mim install mmdet==3.3.0
mim install mmengine==0.10.3
mim install mmyolo==0.6.0
```

### 5. 在项目目录/third_party下，mmyolo是空的，进去clone mmyolo并且安装

```bash
git clone https://github.com/open-mmlab/mmyolo.git
pip install -r requirements/albu.txt
mim install -v -e .
```
### 6. 回到项目目录，安装supervision和transformers

```bash
pip install supervision
pip install transformers
```
### 7. 电脑打开科学上网，打开电脑设置>网络和Internet>代理>手动设置代理>编辑代理服务器
在image_demo.py中设置端口

```bash
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
```
其中 http://proxy_ip_address:port 中的 proxy_ip_address 和 port为开启科学上网后的地址和端口



## Demo

See [`demo`](./demo) for more details

- [x] `gradio_demo.py`: Gradio demo, ONNX export
- [x] `image_demo.py`: inference with images or a directory of images
- [x] `simple_demo.py`: a simple demo of YOLO-World, using `array` (instead of path as input).
- [x] `video_demo.py`: inference YOLO-World on videos.
- [x] `inference.ipynb`: jupyter notebook for YOLO-World.

