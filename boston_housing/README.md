## 波士顿房价预测

### 步骤

1. 克隆存储库并打开下载的文件夹。

 ```	
git clone https://github.com/huangzh10/iResearch2.git
cd iResearch2/boston_housing
```

2. 安装必要的 Python 依赖包


	对于 __Mac/OSX__：
	
	```bash
	conda env create -f requirements/iResearch-mac.yml
	source activate iResearch
	```

	对于 __Windows__：
	
	```bash
	conda env create -f requirements/iResearch-windows.yml
	source activate iResearch
	```
3. 打开 notebook

 ```
jupyter notebook boston_housing.ipynb
```

__注意：__ 我们虽然已经实现了一些代码，让你更快地开始工作，你仍需要实现额外的功能，以回答 notebook 中所有的问题。
__除非有要求，否则不要修改任何已经包含的代码。__

### 数据

经过编辑的波士顿房价数据集有490个数据点，每个点有三个特征。这个数据集编辑自[加州大学欧文分校机器学习数据集库（数据集已下线）](https://archive.ics.uci.edu/ml/datasets.html)。

**特征**

1. `RM`: 住宅平均房间数量
2. `LSTAT`: 区域中被认为是低收入阶层的比率
3. `PTRATIO`: 镇上学生与教师数量比例

**目标变量**

`MEDV`: 房屋的中值价格
