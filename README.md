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
    <td align="center">49.3</td>
    <td align="center">82.4</td>
    <td align="center">88.0</td>
    <td align="center">41.4</td>
    <td align="center">68.8</td>
    <td align="center">99.0</td>
    <td align="center">38.0</td>
    <td align="center">80.8</td>
    <td align="center">94.1</td>
    <td align="center">41.3</td>
    <td align="center">82.8</td>
    <td align="center">93.6</td>
    <td align="center">32.0</td>
    <td align="center">52.7</td>
    <td align="center">68.5</td>
    <td align="center">77.1</td>
    <td align="center">78.0</td>
    <td align="center">100.0</td>
    <td align="center">33.8</td>
    <td align="center">67.7</td>
    <td align="center">78.2</td>
    <td align="center">45.0</td>
    <td align="center">83.9</td>
    <td align="center">90.0</td>
    <td align="center">41.2</td>
    <td align="center">69.6</td>
    <td align="center">83.8</td>
    <td align="center">41.0</td>
    <td align="center">87.0</td>
    <td align="center">96.5</td>
    <td align="center">30.2</td>
    <td align="center">51.8</td>
    <td align="center">61.7</td>
    <td align="center">64.4</td>
    <td align="center">82.2</td>
    <td align="center">84.8</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">SimCLR</td>
    <td align="center">46.5</td>
    <td align="center">83.8</td>
    <td align="center">94.5</td>
    <td align="center">46.4</td>
    <td align="center">85.3</td>
    <td align="center">95.3</td>
    <td align="center">41.0</td>
    <td align="center">76.8</td>
    <td align="center">89.6</td>
    <td align="center">51.8</td>
    <td align="center">87.0</td>
    <td align="center">94.2</td>
    <td align="center">29.8</td>
    <td align="center">49.4</td>
    <td align="center">58.3</td>
    <td align="center">32.0</td>
    <td align="center">80.1</td>
    <td align="center">90.3</td>
    <td align="center">39.1</td>
    <td align="center">80.5</td>
    <td align="center">91.8</td>
    <td align="center">44.8</td>
    <td align="center">83.8</td>
    <td align="center">94.6</td>
    <td align="center">38.0</td>
    <td align="center">61.7</td>
    <td align="center">72.6</td>
    <td align="center">37.6</td>
    <td align="center">82.5</td>
    <td align="center">93.8</td>
    <td align="center">29.1</td>
    <td align="center">50.4</td>
    <td align="center">63.8</td>
    <td align="center">36.6</td>
    <td align="center">76.3</td>
    <td align="center">88.4</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">SimSiam</td>
    <td align="center">41.6</td>
    <td align="center">81.7</td>
    <td align="center">96.8</td>
    <td align="center">39.9</td>
    <td align="center">80.8</td>
    <td align="center">90.2</td>
    <td align="center">34.1</td>
    <td align="center">69.2</td>
    <td align="center">85.3</td>
    <td align="center">44.2</td>
    <td align="center">88.0</td>
    <td align="center">96.2</td>
    <td align="center">31.4</td>
    <td align="center">52.4</td>
    <td align="center">52.5</td>
    <td align="center">77.2</td>
    <td align="center">92.4</td>
    <td align="center">92.8</td>
    <td align="center">42.5</td>
    <td align="center">68.9</td>
    <td align="center">86.4</td>
    <td align="center">42.3</td>
    <td align="center">85.3</td>
    <td align="center">97.5</td>
    <td align="center">30.7</td>
    <td align="center">60.5</td>
    <td align="center">66.0</td>
    <td align="center">45.0</td>
    <td align="center">76.9</td>
    <td align="center">92.7</td>
    <td align="center">30.7</td>
    <td align="center">52.6</td>
    <td align="center">52.6</td>
    <td align="center">77.3</td>
    <td align="center">84.8</td>
    <td align="center">99.9</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">SoftTriple</td>
    <td align="center">54.4</td>
    <td align="center">90.2</td>
    <td align="center">96.5</td>
    <td align="center">57.6</td>
    <td align="center">90.9</td>
    <td align="center">97.1</td>
    <td align="center">50.7</td>
    <td align="center">81.2</td>
    <td align="center">88.6</td>
    <td align="center">68.9</td>
    <td align="center">91.9</td>
    <td align="center">96.6</td>
    <td align="center">55.8</td>
    <td align="center">84.7</td>
    <td align="center">91.9</td>
    <td align="center">46.7</td>
    <td align="center">92.0</td>
    <td align="center">98.6</td>
    <td align="center">57.2</td>
    <td align="center">81.3</td>
    <td align="center">88.0</td>
    <td align="center">62.5</td>
    <td align="center">91.8</td>
    <td align="center">97.1</td>
    <td align="center">56.6</td>
    <td align="center">80.4</td>
    <td align="center">87.4</td>
    <td align="center">65.8</td>
    <td align="center">95.7</td>
    <td align="center">98.8</td>
    <td align="center">58.1</td>
    <td align="center">87.3</td>
    <td align="center">95.0</td>
    <td align="center">78.2</td>
    <td align="center">87.3</td>
    <td align="center">90.4</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">ProxyAnchor</td>
    <td align="center">56.8</td>
    <td align="center">90.8</td>
    <td align="center">97.0</td>
    <td align="center">58.4</td>
    <td align="center">90.3</td>
    <td align="center">96.2</td>
    <td align="center">50.1</td>
    <td align="center">73.9</td>
    <td align="center">81.7</td>
    <td align="center">72.7</td>
    <td align="center">92.8</td>
    <td align="center">95.8</td>
    <td align="center">44.7</td>
    <td align="center">75.2</td>
    <td align="center">86.5</td>
    <td align="center">72.3</td>
    <td align="center">96.5</td>
    <td align="center">98.7</td>
    <td align="center">52.8</td>
    <td align="center">72.3</td>
    <td align="center">79.3</td>
    <td align="center">58.9</td>
    <td align="center">90.6</td>
    <td align="center">96.0</td>
    <td align="center">54.8</td>
    <td align="center">74.6</td>
    <td align="center">83.4</td>
    <td align="center">66.5</td>
    <td align="center">94.1</td>
    <td align="center">98.8</td>
    <td align="center">39.1</td>
    <td align="center">55.2</td>
    <td align="center">70.0</td>
    <td align="center">78.5</td>
    <td align="center">83.1</td>
    <td align="center">84.8</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">ZsCo</td>
    <td align="center"><b>48.6</b></td>
    <td align="center"><b>83.7</b></td>
    <td align="center"><b>93.1</b></td>
    <td align="center"><b>50.3</b></td>
    <td align="center"><b>86.4</b></td>
    <td align="center"><b>94.6</b></td>
    <td align="center"><b>45.9</b></td>
    <td align="center"><b>79.8</b></td>
    <td align="center"><b>89.0</b></td>
    <td align="center"><b>66.1</b></td>
    <td align="center"><b>88.7</b></td>
    <td align="center"><b>94.0</b></td>
    <td align="center"><b>42.5</b></td>
    <td align="center"><b>62.3</b></td>
    <td align="center"><b>73.0</b></td>
    <td align="center"><b>50.4</b></td>
    <td align="center"><b>92.7</b></td>
    <td align="center"><b>97.1</b></td>
    <td align="center"><b>47.9</b></td>
    <td align="center"><b>80.6</b></td>
    <td align="center"><b>89.2</b></td>
    <td align="center"><b>54.7</b></td>
    <td align="center"><b>91.0</b></td>
    <td align="center"><b>97.3</b></td>
    <td align="center"><b>49.7</b></td>
    <td align="center"><b>77.3</b></td>
    <td align="center"><b>85.6</b></td>
    <td align="center"><b>58.1</b></td>
    <td align="center"><b>90.2</b></td>
    <td align="center"><b>96.2</b></td>
    <td align="center"><b>47.1</b></td>
    <td align="center"><b>70.9</b></td>
    <td align="center"><b>81.9</b></td>
    <td align="center"><b>60.5</b></td>
    <td align="center"><b>90.4</b></td>
    <td align="center"><b>95.7</b></td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
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
    <td align="center">9.3</td>
    <td align="center">29.9</td>
    <td align="center">48.2</td>
    <td align="center">11.0</td>
    <td align="center">34.2</td>
    <td align="center">53.8</td>
    <td align="center">13.0</td>
    <td align="center">31.3</td>
    <td align="center">47.6</td>
    <td align="center">9.4</td>
    <td align="center">34.6</td>
    <td align="center">52.8</td>
    <td align="center">14.6</td>
    <td align="center">43.9</td>
    <td align="center">62.1</td>
    <td align="center">12.3</td>
    <td align="center">40.5</td>
    <td align="center">59.1</td>
    <td align="center">13.9</td>
    <td align="center">38.5</td>
    <td align="center">54.2</td>
    <td align="center">14.4</td>
    <td align="center">37.5</td>
    <td align="center">53.3</td>
    <td align="center">15.2</td>
    <td align="center">42.5</td>
    <td align="center">59.6</td>
    <td align="center">14.3</td>
    <td align="center">37.1</td>
    <td align="center">54.6</td>
    <td align="center">21.2</td>
    <td align="center">52.0</td>
    <td align="center">69.2</td>
    <td align="center">20.8</td>
    <td align="center">44.3</td>
    <td align="center">58.1</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">SimCLR</td>
    <td align="center">13.4</td>
    <td align="center">34.8</td>
    <td align="center">51.8</td>
    <td align="center">13.7</td>
    <td align="center">37.6</td>
    <td align="center">55.9</td>
    <td align="center">13.6</td>
    <td align="center">35.5</td>
    <td align="center">47.8</td>
    <td align="center">12.5</td>
    <td align="center">38.6</td>
    <td align="center">57.0</td>
    <td align="center">13.0</td>
    <td align="center">41.2</td>
    <td align="center">55.9</td>
    <td align="center">13.4</td>
    <td align="center">41.9</td>
    <td align="center">58.7</td>
    <td align="center">18.3</td>
    <td align="center">42.4</td>
    <td align="center">57.6</td>
    <td align="center">16.1</td>
    <td align="center">44.1</td>
    <td align="center">60.3</td>
    <td align="center">23.0</td>
    <td align="center">50.2</td>
    <td align="center">65.5</td>
    <td align="center">16.9</td>
    <td align="center">43.8</td>
    <td align="center">59.3</td>
    <td align="center">25.2</td>
    <td align="center">56.0</td>
    <td align="center">69.9</td>
    <td align="center">21.8</td>
    <td align="center">46.0</td>
    <td align="center">60.7</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">SimSiam</td>
    <td align="center">13.8</td>
    <td align="center">36.7</td>
    <td align="center">51.1</td>
    <td align="center">15.4</td>
    <td align="center">41.9</td>
    <td align="center">58.6</td>
    <td align="center">9.1</td>
    <td align="center">32.0</td>
    <td align="center">47.8</td>
    <td align="center">9.6</td>
    <td align="center">35.1</td>
    <td align="center">54.9</td>
    <td align="center">13.2</td>
    <td align="center">38.4</td>
    <td align="center">57.7</td>
    <td align="center">12.1</td>
    <td align="center">39.2</td>
    <td align="center">56.8</td>
    <td align="center">16.7</td>
    <td align="center">39.9</td>
    <td align="center">55.6</td>
    <td align="center">15.1</td>
    <td align="center">42.4</td>
    <td align="center">56.2</td>
    <td align="center">19.3</td>
    <td align="center">45.8</td>
    <td align="center">61.7</td>
    <td align="center">13.7</td>
    <td align="center">38.2</td>
    <td align="center">53.5</td>
    <td align="center">20.3</td>
    <td align="center">49.6</td>
    <td align="center">66.3</td>
    <td align="center">18.2</td>
    <td align="center">42.1</td>
    <td align="center">59.1</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">SoftTriple</td>
    <td align="center">26.2</td>
    <td align="center">55.1</td>
    <td align="center">70.7</td>
    <td align="center">27.7</td>
    <td align="center">60.7</td>
    <td align="center">73.5</td>
    <td align="center">26.0</td>
    <td align="center">52.8</td>
    <td align="center">64.3</td>
    <td align="center">28.4</td>
    <td align="center">59.9</td>
    <td align="center">76.7</td>
    <td align="center">32.6</td>
    <td align="center">65.6</td>
    <td align="center">74.8</td>
    <td align="center">37.1</td>
    <td align="center">65.3</td>
    <td align="center">79.7</td>
    <td align="center">33.2</td>
    <td align="center">54.7</td>
    <td align="center">65.9</td>
    <td align="center">37.5</td>
    <td align="center">62.9</td>
    <td align="center">74.7</td>
    <td align="center">39.6</td>
    <td align="center">67.9</td>
    <td align="center">78.5</td>
    <td align="center">39.4</td>
    <td align="center">66.1</td>
    <td align="center">77.9</td>
    <td align="center">53.8</td>
    <td align="center">80.0</td>
    <td align="center">87.1</td>
    <td align="center">46.5</td>
    <td align="center">71.2</td>
    <td align="center">81.8</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">ProxyAnchor</td>
    <td align="center">27.8</td>
    <td align="center">53.4</td>
    <td align="center">65.8</td>
    <td align="center">29.1</td>
    <td align="center">61.5</td>
    <td align="center">76.2</td>
    <td align="center">24.1</td>
    <td align="center">50.1</td>
    <td align="center">64.9</td>
    <td align="center">30.6</td>
    <td align="center">62.1</td>
    <td align="center">77.2</td>
    <td align="center">30.5</td>
    <td align="center">61.0</td>
    <td align="center">76.3</td>
    <td align="center">33.3</td>
    <td align="center">65.8</td>
    <td align="center">81.1</td>
    <td align="center">32.8</td>
    <td align="center">55.8</td>
    <td align="center">68.2</td>
    <td align="center">40.7</td>
    <td align="center">63.9</td>
    <td align="center">74.9</td>
    <td align="center">40.1</td>
    <td align="center">68.1</td>
    <td align="center">78.4</td>
    <td align="center">35.3</td>
    <td align="center">60.4</td>
    <td align="center">75.3</td>
    <td align="center">51.6</td>
    <td align="center">79.8</td>
    <td align="center">88.8</td>
    <td align="center">45.8</td>
    <td align="center">69.8</td>
    <td align="center">79.2</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">ZsCo</td>
    <td align="center"><b>18.6</b></td>
    <td align="center"><b>43.3</b></td>
    <td align="center"><b>58.8</b></td>
    <td align="center"><b>19.4</b></td>
    <td align="center"><b>45.5</b></td>
    <td align="center"><b>59.8</b></td>
    <td align="center"><b>15.7</b></td>
    <td align="center"><b>39.0</b></td>
    <td align="center"><b>53.4</b></td>
    <td align="center"><b>15.1</b></td>
    <td align="center"><b>44.2</b></td>
    <td align="center"><b>61.6</b></td>
    <td align="center"><b>18.8</b></td>
    <td align="center"><b>49.3</b></td>
    <td align="center"><b>65.6</b></td>
    <td align="center"><b>18.4</b></td>
    <td align="center"><b>47.0</b></td>
    <td align="center"><b>63.9</b></td>
    <td align="center"><b>22.5</b></td>
    <td align="center"><b>48.7</b></td>
    <td align="center"><b>59.5</b></td>
    <td align="center"><b>27.3</b></td>
    <td align="center"><b>52.3</b></td>
    <td align="center"><b>64.0</b></td>
    <td align="center"><b>28.3</b></td>
    <td align="center"><b>55.9</b></td>
    <td align="center"><b>70.8</b></td>
    <td align="center"><b>22.5</b></td>
    <td align="center"><b>48.0</b></td>
    <td align="center"><b>64.3</b></td>
    <td align="center"><b>33.9</b></td>
    <td align="center"><b>62.6</b></td>
    <td align="center"><b>74.3</b></td>
    <td align="center"><b>30.1</b></td>
    <td align="center"><b>55.4</b></td>
    <td align="center"><b>67.6</b></td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
</tbody>
</table>

### T-SNE

![tsne](result/tsne.png)
