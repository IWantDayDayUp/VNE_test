# os.path 模块

import os, sys

def main():

    # 返回绝对路径: d:\VNE_test\notebooks\os.path模块.py
    print("绝对路径: ", os.path.abspath(__file__))
    
    # 返回path的真实路径: D:\VNE_test\notebooks\os.path模块.py
    print("返回path的真实路径: ", os.path.realpath(__file__))
    
    # 返回文件名: os.path模块.py
    print("文件名: ", os.path.basename(__file__))
    
    # 返回文件路径: d:\VNE_test\notebooks
    print("文件路径: ", os.path.dirname(__file__))
    
    # 把目录和文件名合成一个路径: d:\VNE_test\notebooks\os.path模块.py
    print("把目录和文件名合成一个路径: ", os.path.join(os.path.dirname(__file__), os.path.basename(__file__)))

    # 把路径分割成 dirname 和 basename, 返回一个元组: ('d:\\VNE_test\\notebooks', 'os.path模块.py')
    print("把路径分割成 dirname 和 basename: ", os.path.split(os.path.abspath(__file__)))

    # 如果路径 path 存在, 返回 True; 如果路径 path 不存在, 返回 False
    print("判断文件/目录是否存在: ", os.path.exists(os.path.abspath(__file__)))
    
    # 判断路径是否为文件
    print("判断路径是否为文件: ", os.path.isfile(os.path.abspath(__file__)))
    
    # 判断路径是否为目录
    print("判断路径是否为目录: ", os.path.isdir(os.path.abspath(__file__)))

if __name__ == "__main__":
    main()