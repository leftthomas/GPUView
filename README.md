# CenterMask
A PyTorch implementation of CenterMask based on the paper [CenterMask: single shot instance segmentation with point representation](https://arxiv.org/abs/2004.04446).

![Network Architecture image from the paper](structure.png)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
- detectron2
```
pip install git+https://github.com/facebookresearch/detectron2.git@master
```
- opencv
```
pip install opencv-python
```

## Datasets
`MS COCO 2017` dataset is used in this repo. The dataset is assumed to exist in a directory specified by the 
environment variable `DETECTRON2_DATASETS`. You can set the location for this dataset by 
`export DETECTRON2_DATASETS=/path/to/datasets`. The dataset structure should be organized as 
[this](https://github.com/facebookresearch/detectron2/tree/master/datasets) described.

## Training
To train a model, run
```bash
python train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end CenterMask with `DLA-34` backbone training on 8 GPUs, one should execute:
```bash
python train_net.py --config-file configs/dla_34.yaml --num-gpus 8
```

## Evaluation
Model evaluation can be done similarly:
```bash
python train_net.py --config-file configs/dla_34.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS epochs/model.pth
```

## Visualization
Visualize model output can be done like this:
```bash
python visualize.py --input output/dla_34_out.json --output results --dataset coco_2017_minival
```

## Results
There are some difference between this implementation and official implementation:
1. Not support the `Hourglass-104` backbone;
2. The test image sizes are not resized to `600*1000`;
3. The training step is `270,000` for `coco` dataset and `18,000` for `voc` dataset;
4. The `label embedding` of `semantic relation network` is obtained by `mean` of two word embeddings
 of some class labels, such as `traffic light`. 

<table>
	<tbody>
		<th>Method</th>
		<th>AP</th>
		<th>AP<sup>50</sup></th>
		<th>AP<sup>75</sup></th>
		<th>AP<sup>S</sup></th>
		<th>AP<sup>M</sup></th>
		<th>AP<sup>L</sup></th>
		<th>FPS</th>
		<th>Download</th>
		<tr>
			<td align="center"><a href="configs/dla_34.yaml">DLA-34</a></td>
			<td align="center">50.70</td>
			<td align="center">71.80</td>
			<td align="center">56.01</td>
		    <td align="center">30.22</td>
			<td align="center">57.83</td>
			<td align="center">68.73</td>
			<td align="center">25.2</td>
			<td align="center"><a href="https://pan.baidu.com/s/1jP7zWezVPBZWx_9LjJCgWg">xxi8</a></td>
		</tr>
		<tr>
			<td align="center"><a href="configs/hourglass_104.yaml">Hourglass-104</a></td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center"><a href="https://pan.baidu.com/s/1BeGS7gckGAczd1euB55EFA">1jhd</a></td>
		</tr>
	</tbody>
</table>
