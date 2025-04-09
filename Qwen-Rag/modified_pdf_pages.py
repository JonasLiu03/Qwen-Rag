from PyPDF2 import PdfReader, PdfWriter

# 输入文件路径
input_pdf = "/mnt/workspace/LLaMA-Factory/paper/HLLM.pdf"
# 输出文件路径
output_pdf = "/mnt/workspace/LLaMA-Factory/paper/HLLM_first_7_pages.pdf"

# 创建 PdfReader 和 PdfWriter 对象
reader = PdfReader(input_pdf)
writer = PdfWriter()

# 提取前 7 页
for page_num in range(min(7, len(reader.pages))):  # 确保不超过实际页数
    writer.add_page(reader.pages[page_num])

# 写入到输出文件
with open(output_pdf, "wb") as output_file:
    writer.write(output_file)

print(f"成功提取前 7 页，保存为 {output_pdf}")