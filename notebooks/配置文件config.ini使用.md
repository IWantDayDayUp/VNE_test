# 配置文件 config.ini 的使用

## 1. 基本结构

ini最常见一种配置文件类型, 其使用python自带的configparser模块实现配置文件的写入, 更新, 删除, 读取等

使用前请安装 **configparser** 模块

```cmd
pip install configparser
```

基本结构:

```ini
[section_name]
key_name: value
```

默认以 `字符串` 存储, 因此 **value** 不需要加 **双引号**, 例如: `num_nodes: 50`

## 2. 简单使用

- 导入 **configparser** 模块

```python
import configparser
```

- 实例化

```python
config = configparser.ConfigParser()
```

- 读取

```python

# 读取config.ini文件
config.read(filepath, encoding="utf-8")

# 返回ini文件中所有的section的list
config.sections()

# 获取指定section的key list
config.options(section_name)

# 获取指定section的键值对
config.items(section_name)

# 获取指定section下指定key的value值, 返回 str 类型
config.get(section_name, key_name)

```
