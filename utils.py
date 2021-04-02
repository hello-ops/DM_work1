import os
import matplotlib.pyplot as plt 
import scipy.stats as stats
 
 
 # 绘制直方图, 正态分布图，盒图
def draw_figure(data, title, write_path):
    # 绘制直方图
    draw_path = os.path.join(write_path, title+"_直方图.png")
    draw_histogram(data, title, draw_path)

    # 绘制正太分布图
    draw_path = os.path.join(write_path, title+"_正态图.png")
    draw_qq(data, title, draw_path)

    # 绘制盒图
    draw_path = os.path.join(write_path, title+"_盒图.png")
    draw_box_figure(data, title, draw_path)
    
# 绘制直方图
def draw_histogram(data, title, write_path):
    data = data.dropna(subset = [title])
    plt.hist(data[title], 20)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Freq")
    plt.savefig(write_path)
    plt.close()
    
# 绘制正态分布图
def draw_qq(data, title, write_path):
    data = data.dropna(subset = [title])
    stats.probplot(data[title], dist="norm", plot=plt)
    plt.title(title)
    plt.savefig(write_path)
    plt.close()
    
# 绘制盒图
def draw_box_figure(data, title, write_path):
    data = data.dropna(subset=[title])
    a = plt.figure().add_subplot(111)
    a.boxplot(data[title], sym="o", labels=[title], whis=1.5)
    plt.savefig(write_path)
    plt.close()