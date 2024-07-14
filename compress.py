import os
from PIL import Image
from PIL import ImageFile
import argparse
 
# 压缩图片文件
def compress_image(outfile, mb=100, quality=85, k=0.9): # 通常你只需要修改mb大小
    """不改变图片尺寸压缩到指定大小
    :param outfile: 压缩文件保存地址
    :param mb: 压缩目标，KB
    :param k: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件地址，压缩文件大小
    """
 
    o_size = os.path.getsize(outfile) // 1024  # 函数返回为字节，除1024转为kb（1kb = 1024 bit）
    print('before_size:{} after_size:{}'.format(o_size, mb))
    if o_size <= mb:
        return outfile
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # 防止图像被截断而报错
    
    while o_size > mb:
        im = Image.open(outfile)
        x, y = im.size
        out = im.resize((int(x*k), int(y*k)), Image.LANCZOS)  # 最后一个参数设置可以提高图片转换后的质量
        try:
            out.save(outfile, quality=quality)  # quality为保存的质量，从1（最差）到95（最好），此时为85
        except Exception as e:
            print(e)
            break
        o_size = os.path.getsize(outfile) // 1024
    return outfile


def process_dir(directory, mb=100, quality=85, k=0.9):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            compress_image(filepath)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Image dir to be compressed.")
    parser.add_argument(
        "--compress_dir",
        type=str
    )
    parser.add_argument(
        "--target_size_kb",
        type=int,
        default=100
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    process_dir(args.compress_dir, args.target_size_kb)