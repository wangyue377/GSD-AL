import os
import random


def main():
    # random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = r"E:\DOIR_R\test\images"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    # val_rate = 0.5

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    # val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    test_files = []
    # val_files = []
    for index, file_name in enumerate(files_name):
        test_files.append(file_name)

    try:
        test_f = open("test.txt", "x")
        test_f.write("\n".join(test_files))

    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
