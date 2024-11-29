# 标签文件格式

## json格式标签

标签文件中，每一条的格式为`数据路径:标签`，标签可以是任何一种json对象
```json
{
  "data_path": "标签",
  ...
}
```

标签也可以是一个dict，一个图片有多个维度的标注
```json
{
  "data_path": {
    "class": "类别",
    "domain": "域"
  },
  ...
}
```

## yaml格式标签

yaml格式的标签与json格式的内容一样
```yaml
data_path: "标签"
```

```yaml
data_path:
  class: "类别",
  domain: "域"
```

## txt格式标签
txt格式的标签每条数据文件都对应一个。比如对图片`data1.png`，它的标注就是相同文件夹下的`data1.txt`。

```{warning}
这种格式的标签只能是一个字符串，不能是dict等类型。
```