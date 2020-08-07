# Relation R-CNN
A PyTorch implementation of Relation R-CNN based on the paper [Improving Object Detection with Relation Mining Network]().

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
- torchtext
```
conda install -c pytorch torchtext
```

## Datasets
`PASCAL VOC 2007` and `MS COCO 2015` datasets are used in this repo. The datasets are assumed to exist in a directory 
specified by the environment variable `DETECTRON2_DATASETS`. You can set the location for these datasets by 
`export DETECTRON2_DATASETS=/path/to/datasets`. The dataset structure should be organized as 
[this](https://github.com/facebookresearch/detectron2/tree/master/datasets) described.

## Training
To train a model, run
```bash
python train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end Relation R-CNN training for `coco` dataset on 8 GPUs, one should execute:
```bash
python train_net.py --config-file configs/relation_rcnn_coco.yaml --num-gpus 8
```

## Evaluation
Model evaluation can be done similarly:
```bash
python train_net.py --config-file configs/relation_rcnn_coco.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS epochs/model.pth
```

## Visualization
Visualize model output can be done like this:
```bash
python visualize.py --input output/relation_rcnn_coco_out.json --output results --dataset coco_2014_minival
```

## Results
There are some difference between this implementation and official implementation:
1. Not support the `VGG16` backbone;
2. The test image sizes are not resized to `600*1000`;
3. The training step is `270,000` for `coco` dataset and `18,000` for `voc` dataset;
4. The `label embedding` of `semantic relation network` is obtained by `mean` of two word embeddings
 of some class labels, such as `traffic light`. 

### VOC
<table>
	<tbody>
		<th>Method</th>
		<th>AP %</th>
		<th>AP.50 %</th>
		<th>AP.75 %</th>
		<th>download link</th>
		<tr>
			<td align="center"><a href="configs/faster_rcnn_voc.yaml">Faster R-CNN</a></td>
			<td align="center">55.62</td>
			<td align="center">81.81</td>
			<td align="center">61.53</td>
			<td align="center"><a href="https://pan.baidu.com/s/1jP7zWezVPBZWx_9LjJCgWg">model</a>&nbsp;|&nbsp;xxi8</td>
		</tr>
		<tr>
			<td align="center"><a href="configs/relation_rcnn_voc.yaml">Relation R-CNN</a></td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center"><a href="https://pan.baidu.com/s/1BeGS7gckGAczd1euB55EFA">model</a>&nbsp;|&nbsp;1jhd</td>
		</tr>
	</tbody>
</table>

### COCO
<table>
	<tbody>
		<th>Method</th>
		<th>AP %</th>
		<th>AP.50 %</th>
		<th>AP.75 %</th>
		<th>download link</th>
		<tr>
			<td align="center"><a href="configs/faster_rcnn_coco.yaml">Faster R-CNN</a></td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center"><a href="https://pan.baidu.com/s/1jP7zWezVPBZWx_9LjJCgWg">model</a>&nbsp;|&nbsp;xxi8</td>
		</tr>
		<tr>
			<td align="center"><a href="configs/relation_rcnn_coco.yaml">Relation R-CNN</a></td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center"><a href="https://pan.baidu.com/s/1BeGS7gckGAczd1euB55EFA">model</a>&nbsp;|&nbsp;1jhd</td>
		</tr>
	</tbody>
</table>
