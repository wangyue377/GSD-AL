import os
import xml.etree.ElementTree as ET


def get_file_order(xml_file_path):
    """从文件名中提取序号（假设文件名格式为类似00016.jpg）"""
    filename = os.path.basename(xml_file_path)
    name_without_ext = os.path.splitext(filename)[0]
    # 提取纯数字序号（假设文件名全为数字）
    if name_without_ext.isdigit():
        return int(name_without_ext)
    else:
        raise ValueError(f"文件名格式错误，无法提取序号: {filename}")


def convert_xml_to_txt(xml_file_path, output_base_dir, total_trainval=11725):
    """
    将单个XML文件转换为TXT文件，并根据序号分配到trainval/test目录
    :param xml_file_path: XML文件路径
    :param output_base_dir: 输出基础目录（包含trainval和test子目录）
    :param total_trainval: trainval文件夹包含的文件总数
    """
    try:
        # 解析XML文件（仅用于获取文件名，无需处理内容）
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        filename = root.findtext('filename')
        txt_filename = os.path.splitext(filename)[0] + '.txt'

        # 获取文件序号（根据文件名数字顺序）
        order = get_file_order(xml_file_path)

        # 确定输出目录：前total_trainval个为trainval，其余为test
        if order <= total_trainval:
            output_dir = os.path.join(output_base_dir, 'trainval')
        else:
            output_dir = os.path.join(output_base_dir, 'test')

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        txt_file_path = os.path.join(output_dir, txt_filename)

        # 重新解析XML获取数据（避免重复解析）
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        lines = []

        for obj in root.findall('object'):
            robndbox = obj.find('robndbox')
            coords = [
                robndbox.findtext('x_left_top'),
                robndbox.findtext('y_left_top'),
                robndbox.findtext('x_right_top'),
                robndbox.findtext('y_right_top'),
                robndbox.findtext('x_right_bottom'),
                robndbox.findtext('y_right_bottom'),
                robndbox.findtext('x_left_bottom'),
                robndbox.findtext('y_left_bottom')
            ]
            name = obj.findtext('name')
            difficult = obj.findtext('difficult')
            line = ' '.join(coords + [name, difficult])
            lines.append(line)

        # 写入TXT文件
        with open(txt_file_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"成功转换: {xml_file_path} -> {txt_file_path}")

    except Exception as e:
        print(f"处理文件失败 ({xml_file_path}): {str(e)}")


def batch_convert(xml_dir, output_base_dir, total_trainval=11725):
    """
    批量转换XML目录下的所有文件，并按序号分配目录
    :param xml_dir: 输入XML文件目录（需按文件名顺序排列）
    :param output_base_dir: 输出基础目录
    :param total_trainval: trainval文件夹包含的文件总数
    """
    # 获取所有XML文件并按文件名排序（确保序号顺序）
    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith('.xml')],
                       key=lambda x: get_file_order(os.path.join(xml_dir, x)))

    for i, file in enumerate(xml_files, 1):
        xml_path = os.path.join(xml_dir, file)
        convert_xml_to_txt(xml_path, output_base_dir, total_trainval)


if __name__ == "__main__":
    # 配置参数
    INPUT_XML_DIR = r"C:\Users\Administrator\Downloads\Annotations\Oriented Bounding Boxes"  # 输入XML文件所在目录
    OUTPUT_BASE_DIR = r"C:\Users\Administrator\Downloads\output"  # 输出基础目录（自动创建trainval/test子目录）
    TRAINVAL_FILE_COUNT = 11725  # trainval文件夹文件数量

    # 执行批量转换
    batch_convert(INPUT_XML_DIR, OUTPUT_BASE_DIR, TRAINVAL_FILE_COUNT)
    print("批量转换完成！")
