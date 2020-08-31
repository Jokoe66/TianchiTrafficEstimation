# B榜数据下载说明
#### 1.初赛B榜阶段，数据如下:
- amap_traffic_b_test.zip: 初赛B榜测试集视频序列，以文件夹名称为key，每个文件夹中包含一段3-5帧的jpg格式的图像序列,并按照顺序以1-5.jpg的顺序依次存放。
- amap_traffic_annotations_b_test.json: 初赛B榜测试集待提交文件。按照顺序依次存放,以文件夹名称为key，序列的标注为value，格式说明请参考赛题页面描述。其中标注status默认为-1(表示未知),给出各序列关键帧信息。

#### 2.B榜数据的下载链接如下所示：
```bash
https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531809/round1/amap_traffic_b_test_0828.zip # 测试集图片
https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531809/round1/amap_traffic_annotations_b_test_0828.json # 测试集待提交文件
```
#### 3.FAQ：
- 可以将数据链接复制到浏览器中下载，也可使用wget命令直接下载。
- 图像序列文件格式为jpg,标注文件格式为json,使用UTF-8格式编码。后者可通过json.load方法读取，将会获得dict格式的标注信息。
- 上传结果时，应当保证json中所有测试序列信息的完整，否则会得到报错信息：ERROR Not all test sets are included。
- 上传结果的json文件，也应当使用UTF-8编码，按照序列的大小进行排序。