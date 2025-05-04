from PIL import Image
import os

# 要检查的目录路径
directory = '.'  # 当前目录，你可以替换为你想要的目录路径

max_width = 0
max_height = 0

for root, dirs, files in os.walk(directory):
    for file in files:
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:  # 支持的图片格式
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)
                img.close()
            except:
                print(f"无法打开 {file_path}，可能不是有效的图片文件")

print(f"目录下所有图片的最大宽度: {max_width}")
print(f"目录下所有图片的最大高度: {max_height}")
