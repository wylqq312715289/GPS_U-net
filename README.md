### 使用前准备
一、
创建./cache/gen_imgs/目录

二、
将遥感影像原始数据存放到./data/文件夹下
./data/目录下存放文件夹，每个文件夹下要包含遥感影像的tif文件

三、
执行 prepare_imgs.py 将会随机切割遥感影像若干张至./cache/gen_imgs/目录下
人工挑选若干张放入./abelme_gen/imgs/文件目录的下

四、
在控制台运行 labelme 标注./abelme_gen/imgs/下的图像并将.json文件存放入./labelme_gen/jsons/目录下

五、
执行prepare_imgs.py文件中的json_to_dataset方法会在./abelme_gen/jsons/目录下生成对应的标注文件

六、
生成U-net训练数据

注：
main.py 是训练模型的文件，并非处理图像的文件


















































