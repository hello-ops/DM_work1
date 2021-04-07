## 数据探索性分析与数据预处理

```
姓名：刘硕 学号：3220200918
```

## 数据集选择

```
Wine Reviews
Chicago Building Violations
notes：You need to download the required data sets yourself.
	The result file you will see in the other branch(master)
```

## 运行环境及依赖

```
python 3.7.4
pandas, scipy, matplotlib, sklearn
```

## 文件结构

```
hello-ops/DM_work1/
	run.py    # 运行文件
	utils.py
	config.py  # 数据集配置文件
	--data    # 数据集存放
		--wine-reviews
		 --winemag-data_first150k.csv
         --winemag-data-130k-v2
        --Chicago Building Violations
         --building-violations.csv
    --result
    	--wine-reviews
    	 --winemag-data_first150k.csv
    	  --figure                   # 图片结果目录
             --*_his.png
             --*_qq.png
             --*_box.png
          --nominal_attribute        # 标称属性结果目录
             --*.txt
           --numeric_attribute        # 数值属性结果目录
             --*.txt
           --strategy_1               # 缺失值填补结果1目录
             --winemag-data_first150k.csv
             --*_his.png
             --*_qq.png
             --*_box.png
           --strategy_2               # 缺失值填补结果2目录
             --...
           --strategy_3               # 缺失值填补结果3目录
             --...
           --strategy_4               # 缺失值填补结果4目录
             --...
         --winemag-data-130k-v2    
         	--.....
		--building-violations.csv
        	--.....
```

## 运行方法

```
python run.py
# you should modify the config.py files in your environment
data1_path = "wine-reviews"
data1_file_list = ["winemag-data_first150k.csv","winemag-data-130k-v2.csv"]
data2_path = "Chicago Building Violations"
data2_file_list = ["building-violations.csv"]
```

