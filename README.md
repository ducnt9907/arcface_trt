# arcface_trt

First generate arcface_r18.wts from https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

Git clone this repo then put arcface_r18.wts into arcface_trt 

```
cd arcface_trt 

mkdir build

cd build

cmake ..

make 

sudo ./vgg -s // create an engine like 'arcface_r18.engine'
sudo ./vgg -d // run simple test with mtp picture, the embedding vector is written in sth like "mtp_feat.txt" in build/
```
