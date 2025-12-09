https://aistudio.baidu.com/datasetdetail/35331/0

modelnet40_ply_hdf5_2048.zip


原来的数据集路径 这个要改
```cpp
data/modelnet40_ply_hdf5_2048/ply_data_test0.h5
data/modelnet40_ply_hdf5_2048/ply_data_test1.h5
```
改成这样
```cpp
ply_data_test0.h5
ply_data_test1.h5
```


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


pip install tqdm
pip install h5py
