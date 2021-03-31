# ZsCo

A PyTorch implementation of ZsCo based on ACM MM 2021 paper [Zero-shot Contrast Learning for Image Retrieval]().

![Network Architecture](result/structure.jpg)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch
```

- [Pytorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)

```
pip install pytorch-metric-learning
```

- [THOP](https://github.com/Lyken17/pytorch-OpCounter)

```
pip install thop
```

- bidict

```
pip install bidict
```

## Dataset

[PACS](https://domaingeneralization.github.io) and [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
datasets are used in this repo, you could download these datasets from official websites, or download them from
[MEGA](https://mega.nz/folder/M8RFgCzL#nLK35A45QVLCTFqqRzc3vQ). The data should be rearranged, please refer the paper to
acquire the details of `train/val` split. The data directory structure is shown as follows:

 ```
pacs
    ├── art (art images)
        ├── train
            ├── dog
                pic_001.jpg
                ...    
            ...  
        ├── val
            ├── horse
                pic_001.jpg
                ...    
            ...  
    ├── cartoon (cartoon images)
        same structure as art
        ...   
    ...        
office
    same structure as pacs
```

## Usage

```
python main.py or comp.py --data_name office
optional arguments:
# common args
--data_root                   Datasets root path [default value is 'data']
--data_name                   Dataset name [default value is 'pacs'](choices=['pacs', 'office'])
--method_name                 Compared method name [default value is 'zsco'](choices=['zsco', 'simsiam', 'simclr', 'npid', 'proxyanchor', 'softtriple'])
--hidden_dim                  Hidden feature dim for projection head [default value is 512]
--temperature                 Temperature used in softmax [default value is 0.1]
--batch_size                  Number of images in each mini-batch [default value is 32]
--total_iter                  Number of bp to train [default value is 10000]
--ranks                       Selected recall to val [default value is [1, 5, 10]]
--save_root                   Result saved root path [default value is 'result']
# args for zsco
--style_num                   Number of used styles [default value is 8]
--gan_iter                    Number of bp to train gan model [default value is 4000]
--rounds                      Number of round to train whole model [default value is 5]
```

For example, to train `npid` on `office` dataset:

```
python comp.py --method_name npid --data_name office --batch_size 64
```

to train `zsco` on `pacs` dataset, with `16` random selected styles:

```
python main.py --method_name zsco --data_name pacs --style_num 16
```

## Benchmarks

The models are trained on one NVIDIA GTX TITAN (12G) GPU. `Adam` is used to optimize the model, `lr` is `1e-3`
and `weight decay` is `1e-6`. `batch size` is `32` for `zsco`, `simsiam` and `simclr`, `64` for `npid`, `proxyanchor`
and `softtriple`. `lr` is `2e-4` and `betas` is `(0.5, 0.999)` for GAN, other hyper-parameters are the default values.

### PACS

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="3">Art --&gt; Cartoon</th>
    <th colspan="3">Cartoon --&gt; Art</th>
    <th colspan="3">Art --&gt; Photo</th>
    <th colspan="3">Photo --&gt; Art</th>
    <th colspan="3">Art --&gt; Sketch</th>
    <th colspan="3">Sketch --&gt; Art</th>
    <th colspan="3">Cartoon --&gt; Photo</th>
    <th colspan="3">Photo --&gt; Cartoon</th>
    <th colspan="3">Cartoon --&gt; Sketch</th>
    <th colspan="3">Sketch --&gt; Cartoon</th>
    <th colspan="3">Photo --&gt; Sketch</th>
    <th colspan="3">Sketch --&gt; Photo</th>    
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">NPID</td>
    <td align="center">45.5</td>
    <td align="center">81.0</td>
    <td align="center">89.4</td>
    <td align="center">35.4</td>
    <td align="center">66.7</td>
    <td align="center">89.1</td>
    <td align="center">35.0</td>
    <td align="center">69.2</td>
    <td align="center">81.7</td>
    <td align="center">38.6</td>
    <td align="center">85.0</td>
    <td align="center">95.2</td>
    <td align="center">25.7</td>
    <td align="center">42.5</td>
    <td align="center">47.4</td>
    <td align="center">79.1</td>
    <td align="center">84.8</td>
    <td align="center">84.8</td>
    <td align="center">35.0</td>
    <td align="center">71.9</td>
    <td align="center">93.4</td>
    <td align="center">45.0</td>
    <td align="center">79.5</td>
    <td align="center">87.4</td>
    <td align="center">37.2</td>
    <td align="center">50.2</td>
    <td align="center">51.8</td>
    <td align="center">44.1</td>
    <td align="center">91.6</td>
    <td align="center">92.6</td>
    <td align="center">26.2</td>
    <td align="center">39.6</td>
    <td align="center">44.7</td>
    <td align="center">58.6</td>
    <td align="center">84.8</td>
    <td align="center">97.4</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">SimCLR</td>
    <td align="center">48.0</td>
    <td align="center">83.0</td>
    <td align="center">92.2</td>
    <td align="center">41.0</td>
    <td align="center">86.0</td>
    <td align="center">93.0</td>
    <td align="center">35.6</td>
    <td align="center">63.3</td>
    <td align="center">76.4</td>
    <td align="center">46.9</td>
    <td align="center">85.7</td>
    <td align="center">94.3</td>
    <td align="center">30.7</td>
    <td align="center">53.3</td>
    <td align="center">54.5</td>
    <td align="center">70.5</td>
    <td align="center">93.3</td>
    <td align="center">94.0</td>
    <td align="center">39.1</td>
    <td align="center">71.2</td>
    <td align="center">81.6</td>
    <td align="center">40.7</td>
    <td align="center">79.6</td>
    <td align="center">90.3</td>
    <td align="center">36.7</td>
    <td align="center">67.8</td>
    <td align="center">74.4</td>
    <td align="center">50.2</td>
    <td align="center">76.5</td>
    <td align="center">84.1</td>
    <td align="center">30.4</td>
    <td align="center">53.7</td>
    <td align="center">55.0</td>
    <td align="center">42.4</td>
    <td align="center">85.6</td>
    <td align="center">93.0</td>
    <td align="center"><a href="https://pan.baidu.com/s/1aJGLPODKE4cCHLZYDg96jA">4jvm</a></td>
  </tr>
  <tr>
    <td align="center">SimSiam</td>
    <td align="center">44.0</td>
    <td align="center">84.3</td>
    <td align="center">97.6</td>
    <td align="center">40.3</td>
    <td align="center">85.2</td>
    <td align="center">97.0</td>
    <td align="center">36.2</td>
    <td align="center">74.1</td>
    <td align="center">88.0</td>
    <td align="center">40.2</td>
    <td align="center">86.7</td>
    <td align="center">96.9</td>
    <td align="center">31.6</td>
    <td align="center">98.0</td>
    <td align="center">99.5</td>
    <td align="center">58.2</td>
    <td align="center">95.6</td>
    <td align="center">99.7</td>
    <td align="center">32.2</td>
    <td align="center">76.9</td>
    <td align="center">93.8</td>
    <td align="center">42.9</td>
    <td align="center">82.3</td>
    <td align="center">96.7</td>
    <td align="center">38.0</td>
    <td align="center">81.5</td>
    <td align="center">92.8</td>
    <td align="center">29.6</td>
    <td align="center">71.6</td>
    <td align="center">83.2</td>
    <td align="center">29.4</td>
    <td align="center">99.1</td>
    <td align="center">99.7</td>
    <td align="center">69.1</td>
    <td align="center">85.8</td>
    <td align="center">99.6</td>
    <td align="center"><a href="https://pan.baidu.com/s/1aJGLPODKE4cCHLZYDg96jA">4jvm</a></td>
  </tr>
  <tr>
    <td align="center">SoftTriple</td>
    <td align="center">47.0</td>
    <td align="center">87.5</td>
    <td align="center">96.8</td>
    <td align="center">48.7</td>
    <td align="center">89.0</td>
    <td align="center">96.4</td>
    <td align="center">50.8</td>
    <td align="center">81.8</td>
    <td align="center">90.9</td>
    <td align="center">56.5</td>
    <td align="center">90.5</td>
    <td align="center">96.7</td>
    <td align="center">42.8</td>
    <td align="center">73.0</td>
    <td align="center">89.0</td>
    <td align="center">69.5</td>
    <td align="center">95.6</td>
    <td align="center">98.8</td>
    <td align="center">48.8</td>
    <td align="center">84.5</td>
    <td align="center">92.6</td>
    <td align="center">51.5</td>
    <td align="center">87.6</td>
    <td align="center">96.2</td>
    <td align="center">49.6</td>
    <td align="center">78.2</td>
    <td align="center">91.3</td>
    <td align="center">56.4</td>
    <td align="center">90.3</td>
    <td align="center">97.7</td>
    <td align="center">53.5</td>
    <td align="center">78.4</td>
    <td align="center">91.4</td>
    <td align="center">61.2</td>
    <td align="center">93.0</td>
    <td align="center">98.7</td>
    <td align="center"><a href="https://pan.baidu.com/s/1mYIRpX4ABX9YVLs0gFJVmg">6we5</a></td>
  </tr>
  <tr>
    <td align="center">ProxyAnchor</td>
    <td align="center">44.8</td>
    <td align="center">86.3</td>
    <td align="center">96.9</td>
    <td align="center">46.6</td>
    <td align="center">87.3</td>
    <td align="center">95.1</td>
    <td align="center">50.7</td>
    <td align="center">86.6</td>
    <td align="center">94.8</td>
    <td align="center">55.7</td>
    <td align="center">90.3</td>
    <td align="center">96.2</td>
    <td align="center">43.9</td>
    <td align="center">79.5</td>
    <td align="center">90.3</td>
    <td align="center">59.2</td>
    <td align="center">95.3</td>
    <td align="center">98.8</td>
    <td align="center">44.8</td>
    <td align="center">78.6</td>
    <td align="center">88.9</td>
    <td align="center">43.7</td>
    <td align="center">85.1</td>
    <td align="center">95.8</td>
    <td align="center">46.4</td>
    <td align="center">77.4</td>
    <td align="center">86.3</td>
    <td align="center">61.5</td>
    <td align="center">92.3</td>
    <td align="center">97.4</td>
    <td align="center">41.9</td>
    <td align="center">83.2</td>
    <td align="center">93.6</td>
    <td align="center">40.6</td>
    <td align="center">84.1</td>
    <td align="center">95.5</td>
    <td align="center"><a href="https://pan.baidu.com/s/1aEQhoDH3ciAHESbzSfeR6Q">99k3</a></td>
  </tr>
  <tr>
    <td align="center">ZsCo</td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><a href="https://pan.baidu.com/s/19d3v1PTnX-Z3dH7ifeY1oA">cb2b</a></td>
  </tr>
</tbody>
</table>

### Office-Home

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="3">Art --&gt; Clipart</th>
    <th colspan="3">Clipart --&gt; Art</th>
    <th colspan="3">Art --&gt; Product</th>
    <th colspan="3">Product --&gt; Art</th>
    <th colspan="3">Art --&gt; Real</th>
    <th colspan="3">Real --&gt; Art</th>
    <th colspan="3">Clipart --&gt; Product</th>
    <th colspan="3">Product --&gt; Clipart</th>
    <th colspan="3">Clipart --&gt; Real</th>
    <th colspan="3">Real --&gt; Clipart</th>
    <th colspan="3">Product --&gt; Real</th>
    <th colspan="3">Real --&gt; Product</th>    
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
    <td align="center">R@1</td>
    <td align="center">R@5</td>
    <td align="center">R@10</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">NPID</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">SimCLR</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"><a href="https://pan.baidu.com/s/1aJGLPODKE4cCHLZYDg96jA">4jvm</a></td>
  </tr>
  <tr>
    <td align="center">SimSiam</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"><a href="https://pan.baidu.com/s/1aJGLPODKE4cCHLZYDg96jA">4jvm</a></td>
  </tr>
  <tr>
    <td align="center">SoftTriple</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"><a href="https://pan.baidu.com/s/1mYIRpX4ABX9YVLs0gFJVmg">6we5</a></td>
  </tr>
  <tr>
    <td align="center">ProxyAnchor</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"><a href="https://pan.baidu.com/s/1aEQhoDH3ciAHESbzSfeR6Q">99k3</a></td>
  </tr>
  <tr>
    <td align="center">ZsCo</td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><b></b></td>
    <td align="center"><a href="https://pan.baidu.com/s/19d3v1PTnX-Z3dH7ifeY1oA">cb2b</a></td>
  </tr>
</tbody>
</table>

### T-SNE

![tsne](result/tsne.png)
