---
{"dg-publish":true,"permalink":"/pytorch/dataset/","dgPassFrontmatter":true,"created":"2025-07-04T12:51:01.430+08:00","updated":"2025-12-26T14:58:34.062+08:00"}
---

F:\pytorch学习资料\Pytorch学习_小土堆\Dataset_dataloader.py
F:\pytorch学习资料\Pytorch学习_小土堆\Dataset_dataloader.ipynb
## 功能

Dataset 类 一般用于加载 数据集 及其 对应的标签

## 用法 

通过**help(Dataset)** 可以获取如下信息

```python
class Dataset(typing.Generic)
 |  An abstract class representing a :class:`Dataset`.
 |  
 |  All datasets that represent a map from keys to data samples should subclass
 |  it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
 |  data sample for a given key. Subclasses could also optionally overwrite
 |  :meth:`__len__`, which is expected to return the size of the dataset by many
 |  :class:`~torch.utils.data.Sampler` implementations and the default options
 |  of :class:`~torch.utils.data.DataLoader`. Subclasses could also
 |  optionally implement :meth:`__getitems__`, for speedup batched samples
 |  loading. This method accepts list of indices of samples of batch and returns
 |  list of samples.
 |  
 |  .. note::
 |    :class:`~torch.utils.data.DataLoader` by default constructs an index
 |    sampler that yields integral indices.  To make it work with a map-style
 |    dataset with non-integral indices/keys, a custom sampler must be provided.
 |  
 |  Method resolution order:
 |      Dataset
 |      typing.Generic
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]'
 |  
 |  __getitem__(self, index) -> +T_co
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __orig_bases__ = (typing.Generic[+T_co],)
 |
 |  __parameters__ = (+T_co,)
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from typing.Generic:
 |
 |  __class_getitem__(params) from builtins.type
 |
 |  __init_subclass__(*args, **kwargs) from builtins.type
 |      This method is called when a class is subclassed.
 |
 |      The default implementation does nothing. It may be
 |      overridden to extend subclasses.

```

### 应当重写`__getitem__` and `__len__` 且进行初始化定义 `__init__`


### 一个栗子

```python
class MyDataset(Dataset): # 继承这个类
    def __init__(self,root_dir,label_dir):
         self.root_dir = root_dir  # 根目录 视频中为"dataset/train""
         self.label_dir = label_dir  # 视频中直接为“ants”  所以这里直接就是label的名字
         self.path = os.path.join(root_dir, label_dir)
         self.image_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.image_path[index]    ## 这里只是获取图片的名字
        img_item_path = os.path.join(self.path, img_name)  ## 拼接成图片的路径 
        img = Image.open(img_item_path)  ## 打开图片 
        label = self.label_dir
        return img, label
    def __len__(self):
        return len(self.image_path)
    
root_dir = "dataset/train"
label_dir = "ants"
bees_dir = "bees"
ant_dataset = MyDataset(root_dir, label_dir)
bees_dataset = MyDataset(root_dir, bees_dir)

train_dataset = ant_dataset + bees_dataset  # 合并两个数据集   

img,label = ant_dataset[0]  # 获取第一个样本
Image.show(img)  # 显示图片
```