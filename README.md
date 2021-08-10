# arcface_trt

First generate arcface_r18.wts from https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch or download from https://drive.google.com/file/d/1D6Zh_6IgfPXDn1-JJ9o_3EP1MlTx2i3a/view?usp=sharing

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

To compare C++ with Python result, upload "mtp_feat.txt" to https://drive.google.com/drive/folders/1BMc5tGLLfMC_zU901Wbr9LJxD0zYzUQx?usp=sharing and run inference.py in this folder
