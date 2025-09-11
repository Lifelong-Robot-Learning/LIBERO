import h5py
import numpy as np

def explore_hdf5(file_path):
    """
    查看HDF5文件的结构和数据集内容
    """
    try:
        # ✅ 打开HDF5文件，'r'表示只读模式
        with h5py.File(file_path, 'r') as f:
            print("📂 HDF5文件结构:")
            
            # ✅ 递归函数，用于打印文件中的组和数据集
            def print_structure(name, obj):
                indent = '  ' * name.count('/')
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📁 Group: {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}📊 Dataset: {name}")
                    print(f"{indent}  Shape: {obj.shape}")
                    print(f"{indent}  Dtype: {obj.dtype}")
                    # 如果数据集较小，打印部分数据
                    if obj.size < 10:
                        print(f"{indent}  Data: {obj[()]}")
                    else:
                        print(f"{indent}  Data (first 1 elements): {obj[:1]}")
            
            # ✅ 遍历文件结构
            f.visititems(print_structure)
                
    except FileNotFoundError:
        print("❌ 文件不存在，请检查文件路径！")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")


def print_dataset_keys(file_path):
    """
    打印HDF5文件中所有数据集的key（名称路径）
    """
    try:
        # ✅ 打开HDF5文件，'r'表示只读模式
        with h5py.File(file_path, 'r') as f:
            print("📂 数据集的key列表:")
            
            # ✅ 递归函数，仅收集数据集的key
            def collect_dataset_keys(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"📊 {name}")
            
            # ✅ 遍历文件结构，打印数据集key
            f.visititems(collect_dataset_keys)
            
            if not any(isinstance(obj, h5py.Dataset) for _, obj in f.items()):
                print("⚠️ 文件中没有数据集！")
                
    except FileNotFoundError:
        print("❌ 文件不存在，请检查文件路径！")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")


def print_dataset_content(file_path, dataset_key):
    """
    打印HDF5文件中指定数据集key的内容
    :param file_path: HDF5文件路径
    :param dataset_key: 数据集的key（名称路径，如 'demo_1/data'）
    """
    try:
        # ✅ 打开HDF5文件，'r'表示只读模式
        with h5py.File(file_path, 'r') as f:
            # ✅ 检查指定key是否存在
            if dataset_key not in f:
                print(f"❌ 数据集 '{dataset_key}' 不存在！")
                return
            
            # ✅ 获取数据集
            dataset = f[dataset_key]
            if not isinstance(dataset, h5py.Dataset):
                print(f"❌ '{dataset_key}' 不是一个数据集！")
                return
            
            # ✅ 打印数据集信息
            print(f"📊 指定key对应的数据集: {dataset_key}")
            print(f"  Shape: {dataset.shape} ✅")
            print(f"  Dtype: {dataset.dtype} ✅")
            
            # ✅ 打印数据内容
            if dataset.size < 10:
                print(f"  Data: {dataset[()]} 📈")
            else:
                print(f"  Data (first 1 elements): {dataset[:1]} 📈")
                
    except FileNotFoundError:
        print("❌ 文件不存在，请检查文件路径！")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")

# # 示例用法
# if __name__ == "__main__":
#     # 使用你提供的HDF5文件路径
#     file_path = "/home/sythoid_01/文档/Huangwenlong/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
#     # 替换为你想查看的数据集key
#     dataset_key = "data/demo_0/states"  # 示例key，需替换为实际的key
#     print_dataset_content(file_path, dataset_key)

# # 示例用法
# if __name__ == "__main__":
#     # 使用你提供的HDF5文件路径
#     file_path = "/home/sythoid_01/文档/Huangwenlong/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
#     print_dataset_keys(file_path)

# 示例用法
if __name__ == "__main__":
    # 替换为你自己的HDF5文件路径
    file_path = "/home/sythoid_01/文档/Huangwenlong/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    explore_hdf5(file_path)