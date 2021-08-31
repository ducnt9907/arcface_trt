# arcface_trt

First generate wts file from arcface_torch/wts_generator.py (eg:  ``` python wts_generator.py --network='r50' --weight='./backbone_r50.pth' --save_path='./arcface_50.wts' ```)

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

Use arcface_torch/inference.py to evaluate the result (eg: ``` python inference.py --network='r50' --weight='./backbone_r50.pth' --img='./vietth.jpg' --feat='./vietth_feat.txt' ```)
