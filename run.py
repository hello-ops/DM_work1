import os
import pandas as pd
from utils import draw_figure, draw_histogram, draw_qq, draw_box_figure
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from tqdm import tqdm
from config import dataset1_path, dataset1_list, dataset2_list, dataset2_path
class DM():
    def __init__(self):
        self.data_read = [os.path.join("data", dataset1_path), os.path.join("data", dataset2_path)]
        self.data_write = [os.path.join("result", dataset1_path), os.path.join("result", dataset2_path)]
        self.data_list = [dataset1_list, dataset2_list]

    # 对数据进行处理-数据摘要
    def data_summary(self):
        for i in range(len(self.data_list)):
            for name in self.data_list[i]:
                # 通过pandas读取数据
                read_data = pd.read_csv(os.path.join(self.data_read[i], name))
                data_writepath = os.path.join(self.data_write[i], name.split(".")[0])
                print("------------------------------------------")
                print("开始处理数据文件：%s" % name)
                for title in read_data.columns.values: 
                    if title == "Unnamed: 0":
                        continue
                    if read_data[title].dtype == 'int64' or read_data[title].dtypes == 'float64':
                        self.handle_num_data(read_data, title, data_writepath)
                    else:
                        self.handle_oth_data(read_data, title, data_writepath)
    
    # 处理数字数据
    def handle_num_data(self, data, title, write_path):
        write_path = os.path.join(write_path, 'num_attr')
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        
        max_num = data[title].max()
        min_num = data[title].min()
        mean_num = data[title].mean()
        median_num = data[title].median()
        quartile1 = data[title].quantile(0.25)
        quartile2 = data[title].quantile(0.75)
        missing_num = len(data) - data[title].count()
        
        with open(os.path.join(write_path, title+".txt"), "w") as f:
            f.write(("特征名称： %s\n" % title))
            f.write(("最大值： %s\n" % max_num))
            f.write(("最小值： %s\n" % min_num))
            f.write(("均值： %s\n" % mean_num))
            f.write(("中位数: %s\n" % median_num))
            f.write(("分位点: %s, %s\n" % (quartile1, quartile2)))
            f.write(("丢失数量: %s\n" % missing_num))
        print("------------------------------------------")
        print("特征名称： %s\n" % title)
        print("最大值： %s\n" % max_num)
        print("最小值： %s\n" % min_num)
        print("均值： %s\n" % mean_num)
        print("中位数: %s\n" % median_num)
        print("分位点: %s, %s\n" % (quartile1, quartile2))
        print("丢失数量: %s\n" % missing_num)
        print("count: %s\n" % data[title].count())
        print("len: %s\n" % len(data))
        print("------------------------------------------")
        
        write_data_path = os.path.join(write_path, 'figure')
        if not os.path.exists(write_data_path):
            os.makedirs(write_data_path)
        draw_figure(data, title, write_data_path)
    # 处理其他数据类型的数据
    def handle_oth_data(self, data, title, write_path):
        write_path = os.path.join(write_path, 'oth_attr')
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        value_dict = self.get_feature_value(data, title)
        
        with open(os.path.join(write_path, title+".txt"), "w", encoding='utf-8') as f:
            f.write(("attr_name: %s\n" % title))
            f.write(("value_num: %s\n" % len(value_dict)))
            for i in value_dict:
                f.write(str(i) + ": " + str(value_dict[i])+ "\n")
        print("Feature Name: %s" % title)
        print("Value Num: %s" % len(value_dict))
        print("------------------------------------------")
    
    def get_feature_value(self, data, title):
        value_dict = dict()
        for i in range(len(data)):
            if pd.isnull(data[title][i]):
                continue
            if data[title][i] in value_dict:
                value_dict[data[title][i]] += 1
            else:
                value_dict[data[title][i]] = 1
        return value_dict
    
    
    # 对缺失数据进行填补，分别使用四种策略:缺失值剔除，最高频率值来填补缺失值，使用属性的相关关系填
    # 通过数据对象之间的相似性填补
    def fill(self):
        for i in range(len(self.data_list)):
            for name in self.data_list[i]:
                data = pd.read_csv(os.path.join(self.data_read[i], name))
                write_data_path = os.path.join(self.data_write[i], name.split(".")[0])
                print("------------------------------------------")
                print("开始处理数据文件: %s" % name)
                # 策略一 将缺失部分剔除
                strategy_path = os.path.join(write_data_path, "strategy_1")
                if not os.path.exists(strategy_path):
                    os.makedirs(strategy_path)
                with open(os.path.join(strategy_path, name), 'w', encoding='utf-8') as f1:
                    str1_data = data
                    for title in data.columns.values:
                        if title == "Unnamed: 0":
                            str1_data = str1_data.drop(columns = [title])
                        elif str1_data[title].dtypes == "int64" or str1_data[title].dtypes == "np.float64":
                            str1_data = str1_data.dropna(subset=[title])
                            draw_figure(str1_data, title, strategy_path)
                    str1_data.to_csv(f1)
                # 策略二 用最高频率值来填补缺失值
                strategy_path = os.path.join(write_data_path, "strategy_2")
                if not os.path.exists(strategy_path):
                    os.makedirs(strategy_path)
                with open(os.path.join(strategy_path, name), 'w', encoding='utf-8') as f2:
                    str2_data = data
                    for title in data.columns.values:
                        if title == "Unnamed: 0":
                            str2_data = str2_data.drop(columns=[title])
                        elif str2_data[title].dtypes == "int64" or str2_data[title].dtypes == "np.float64":
                            value_dict = self.get_feature_value(str2_data, title)
                            fill_data = max(value_dict, key=value_dict.get)
                            str2_data = str2_data.fillna({title:fill_data})
                            draw_figure(str2_data, title, strategy_path)
                    str2_data.to_csv(f2)
                # 策略三 通过数据对象之间的相似性来填补缺失值
                strategy_path = os.path.join(write_data_path, "strategy_3")
                if not os.path.exists(strategy_path):
                    os.makedirs(strategy_path)
                with open(os.path.join(strategy_path, name), 'w', encoding='utf-8') as f3:
                    str3_data = data
                    str3_data = str3_data.interpolate(kind='nearest')
                    str3_data.to_csv(f3)
                    for title in data.columns.values:
                        if title == "Unnamed: 0":
                            continue
                        elif str3_data[title].dtypes == "int64" or str3_data[title].dtypes == "np.float64":
                            draw_figure(str3_data, title, strategy_path)
                # 策略四
                strategy_path = os.path.join(write_data_path, "strategy_4")
                if not os.path.exists(strategy_path):
                    os.makedirs(strategy_path)
                with open(os.path.join(strategy_path, name), 'w', encoding='utf-8') as f4:
                    str4_data = data
                    nonan_content = pd.DataFrame()
                    num_list = []
                    for title in str4_data.columns.values:
                        if title == "Unnamed: 0":
                            str4_data = str4_data.drop(title, 1)
                        elif str4_data[title].dtypes == "int64" or str4_data[title].dtypes == "float64":
                            num_list.append(title)
                            nonan_content = pd.concat([nonan_content, str4_data[title]], axis=1)
                    nonan_content.dropna(axis=0, how='any', inplace=True)
                    mean_val = [ nonan_content[title].mean() for title in nonan_content.columns.values]

                    if len([title for title in num_list if str4_data[title].isnull().any() == True]) == len(num_list):
                        for i in range(len(str4_data)):
                            if str4_data.loc[i][num_list].isnull().all():
                                for j in range(len(num_list)):
                                    str4_data.loc[i, num_list[j]] = mean_val[j]
                    
                    for title in num_list:
                        if str4_data[title].isnull().any():
                            train_y = nonan_content[title]
                            train_x = nonan_content.loc[:, [other for other in num_list if other != title]]
                            test_x = str4_data[pd.isna(str4_data[title])].loc[:, [other for other in num_list if other != title]]
                            index, pred = self.logis_filled(train_x, train_y, test_x)
                            str4_data.loc[index, title] = pred
                        draw_figure(str4_data, title, strategy_path)
                    str4_data.to_csv(f4)
        
    def logis_filled(self, train_x, train_y, test, k=3, dispersed = True):
        if dispersed:
            clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
        else:
            clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
        clf.fit(train_x, train_y)    
        return test.index, clf.predict(test)

if __name__ == "__main__":
    data = DM()
    # 数据摘要
    data.data_summary()
    # 数据缺失的处理
    data.fill()