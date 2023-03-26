---
title: "Python Guide"
description: "2023 desc"
pubDate: "2023-03-24T12:51:24Z"
heroImage: "/post_img.webp"
badge: "python"

---

# Python Guide

# 🐍Python

## 🔹yield和return

共同点：return和yield都用来返回值；在一次性地返回所有值场景中return和yield的作用是一样的。

不同点：如果要返回的数据是通过for等循环生成的迭代器类型数据（如列表、元组），return只能在循环外部一次性地返回，yeild则可以在循环内部逐个元素返回。下边我们举例说明这个不同点。

https://www.cnblogs.com/andy0816/p/15617462.html

# ✅numpy

## 🔹reshape函数

reshape(1,-1)转化成1行

reshape(2,-1)转换成2行

reshape(-1,2)转化成2列

reshape(2,8)转化成2行8列

# ✅torch

## 🔹repeat函数

```python
import torch
 
x = torch.tensor([1,2,3])
 
#将一维度的x扩展到三维
xx = x.repeat(4,2,1)
 
/**
扩展步骤如下(倒着执行)：
1  最后一个维度1：此时将[1,2,3]中的数字直接重复1次，得到[1,2,3]，保持没变
2  倒数第二个维度2：先将上一步骤的结果增加一个维度，得到[[1,2,3]]，然后将最外层中括号中的整体重复2次，得到[[1,2,3],[1,2,3]]
3  倒数第三个维度4：先将上一步骤的结果增加一个维度，得到[[[1,2,3],[1,2,3]]]，然后将最外层中括号中的整体重复4次，得到[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]
4  三个维度扩展结束，得到结果。
 
**/
```

## 🔹clone函数

返回一个新的tensor，这个tensor与原始tensor的数据不共享一个内存(也就是说， 两者不是同一个数据，修改一个另一个不会变)

requires_grad属性与原始tensor相同，若requires_grad=True，计算梯度，但不会保留梯度，梯度会与原始tensor的梯度相加

## 🔹detach函数

返回一个新的tensor，这个tensor与原始tensor的数据共享一个内存(也就是说，两者是同一个数据，修改原始tensor，new tensor也会变； 修改new tensor，原始tensor也会变)

require_grad设置为False,*截断反向传播的梯度流*

## 🔹eye函数

返回一个对角矩阵

## 🔹fill_diagonal_函数

填充对角线值

## 🔹max，argmax函数

max返回的值有两个，values和indexes。argmax返回的只是indexes

max(tensor, dim) dim=0 按行比较，dim=1 按列比较，

**dim等于0时为求每列的最大值，等于1时为求每行的最大值**

## 🔹stack函数

沿一个新维度对输入张量序列进行连接，序列中所有张量应为相同形状；stack 函数返回的结果会新增一个维度，而stack（）函数指定的dim参数，就是新增维度的（下标）位置

dim：新增维度的（下标）位置，当dim = -1时默认最后一个维度

# 🔧PyCharm

## 🔹debug

|Command|Guide|Operation|
| :--------------------: | :------------------------------------------------------------: | :-----------------: |
|Rerun Window|重新debug此文件|Ctrl + F5|
|Resume Program|放过当前断点，直接跳到下一断点；<br />若无下一断点，直接跑完程序|F9|
|Stop Window|停止当前debug模式，关闭当前程序|Ctrl + F2|
|View Breakpoints|显示所有断点|Ctrl + Shift + F8|
|Mute Breakpoints|使所有断点失效|--|
|Step Into|进入函数中，包括源代码函数|F7|
|Step Into My Code|进入自己写的函数|Alt + Shift + F7|
|Step Over|顺着程序执行代码，不进入函数|F8|
|Step Out|跳出当前函数体|Shift + F8|
|Run To Cursor|运行到光标处|Alt + F9|
|Show Execution Point|跳转到代码当前执行的位置|Alt + F10|

### tensor类型数据的调试

可以**numpy.array(tensor.data.cpu())**来查看

‍
