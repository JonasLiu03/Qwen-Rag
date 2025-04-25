import subprocess
import shlex
import os
import json
import random
import io
import ast
import re
import markdown
from bs4 import BeautifulSoup
import zipfile # 导入 zipfile 库用于处理 zip 文件
import shutil # 导入 shutil 库用于目录操作和清理

from PIL import Image
from openai import OpenAI
import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, List # 导入 List

# --- FastAPI 应用初始化 ---
app = FastAPI(
    title="Document Processing Pipeline API (with Archive Support)", # 更新标题
    description="API to process documents (PDF, TXT, Markdown, Archives, etc.) and extract information.",
    version="1.0.0",
)

# --- 请在这里插入你实际的 magic-pdf 可执行文件的绝对路径 ---
MAGIC_PDF_PATH = "/mnt/workspace/MinerU_Actual/mineru/bin/magic-pdf"
# -------------------------------------------------------------

# --- 在应用启动时检查 magic-pdf 路径 ---
@app.on_event("startup")
async def startup_event():
    if os.path.exists(MAGIC_PDF_PATH):
        print(f"magic-pdf executable found at {MAGIC_PDF_PATH}")
    else:
         print(f"WARNING: magic-pdf executable not found at {MAGIC_PDF_PATH}. PDF and other magic-pdf supported file types will NOT be processed.")


# --- 您的 magic-pdf 执行函数 ---
# 保持与之前提供的最新版一致
def run_magic_pdf(input_pdf_path: str, output_directory: str) -> tuple[int, str, str]:
    # ... (函数体与之前的代码相同) ...
    """
    运行 magic-pdf 命令行工具，使用指定的绝对路径。
    此函数将 output_directory 作为 -o 参数的值传递给 magic-pdf。
    注意：magic-pdf 会在 output_directory 下自动创建子文件夹（例如 InputFileName/auto/）。

    Args:
        input_pdf_path (str): 输入 PDF 文件的完整路径。
        output_directory (str): magic-pdf 输出文件的**根目录**。

    Returns:
        tuple: (return_code, stdout, stderr) 命令行执行结果。
    """
    if not os.path.exists(MAGIC_PDF_PATH):
         return 1, "", f"错误：magic-pdf 路径不可用或不存在: {MAGIC_PDF_PATH}"

    if not os.path.exists(input_pdf_path):
        return 1, "", f"错误：输入文件未找到于 {input_pdf_path}"

    try:
        os.makedirs(output_directory, exist_ok=True)
    except Exception as e:
        return 1, "", f"错误：创建 magic-pdf 输出根目录失败 {output_directory}: {e}"

    command = f"{shlex.quote(MAGIC_PDF_PATH)} -p {shlex.quote(input_pdf_path)} -o {shlex.quote(output_directory)}"

    print(f"API 调用执行命令: {command}")

    try:
        result = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            check=True
        )

        print("命令执行成功。")
        return result.returncode, result.stdout, result.stderr

    except FileNotFoundError:
        error_msg = f"错误：找不到 magic-pdf 命令于 {MAGIC_PDF_PATH}。"
        print(error_msg)
        return 1, "", error_msg
    except subprocess.CalledProcessError as e:
        print(f"错误：命令执行失败，返回码 {e.return_code}") # Changed from e.returncode
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return e.returncode, e.stdout, e.stderr
    except Exception as e:
        error_msg = f"执行命令时发生意外错误: {e}"
        print(error_msg)
        return 1, "", error_msg

# --- 您的 JSON 处理和图片推理代码 ---

# @title Helper function for parsing JSON output
def parse_json(json_output: str) -> Any:
    # ... (函数体与之前的代码相同) ...
    """
    Parses JSON string, removing markdown fencing if present.
    """
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0].strip()
            break
        if line.strip().startswith('{') or line.strip().startswith('['):
             break
    try:
        return json.loads(json_output)
    except json.JSONDecodeError as e:
        print(f"警告: 解析 JSON 失败 {e}. 尝试直接返回原始字符串.")
        return json_output

# @title Helper function for base64 encoding
def encode_image(image_path: str) -> str:
    # ... (函数体与之前的代码相同) ...
    """
    Encodes an image file to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# @title inference function with API
def inference_with_api(image_path: str, prompt: str, sys_prompt: str = "You are a helpful assistant.", model_id: str = "Qwen2.5-VL-7B-Instruct", min_pixels: int = 512*28*28, max_pixels: int = 2048*28*28) -> str:
    # ... (函数体与之前的代码相同) ...
    """
    Inference using an external API (OpenAI compatible).
    """
    print(f"--- Calling inference_with_api for {os.path.basename(image_path)} ---")
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        print(f"错误：编码图片 {image_path} 失败: {e}")
        raise

    llm_cfg = {
        'model': model_id,
        'model_server': 'http://localhost:8000/v1', # 确保这个地址在运行API服务的环境中可访问
        'api_key': 'EMPTY',
        'generate_cfg': {
            'top_p': 0.8,
            'max_tokens': 1024
        }
    }

    try:
        client = OpenAI(
            api_key=llm_cfg['api_key'],
            base_url=llm_cfg['model_server']
        )

        image_format = 'jpeg' # Assume jpeg for now, or determine dynamically

        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": sys_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        completion = client.chat.completions.create(
            model = llm_cfg['model'],
            messages = messages,
        )

        if completion and completion.choices and completion.choices[0].message:
             return completion.choices[0].message.content
        else:
             print("警告：API 调用返回空或非预期结构。")
             return "API 返回空结果"

    except Exception as e:
        print(f"错误：调用 API 进行推理失败: {e}")
        raise

# --- 您的 process_json_with_images 函数 ---
# 保持与之前提供的最新版一致
def process_json_with_images(json_filepath: str) -> tuple[str | None, List[str]]:
    # Modified return type to include errors list
    """
    读取JSON文件，处理图片块，添加img_text字段，并保存修改后的JSON。

    Args:
        json_filepath (str): 输入的JSON文件路径。

    Returns:
        tuple[str | None, List[str]]: 修改后的 JSON 文件路径 (如果成功) 和图片处理错误列表。
                                      如果JSON文件处理失败，返回 (None, errors)。
    """
    if not os.path.exists(json_filepath):
        print(f"错误：process_json_with_images: JSON 文件未找到于 {json_filepath}")
        return None, [f"JSON 文件未找到于 {json_filepath}"]

    json_dir = os.path.dirname(os.path.abspath(json_filepath))
    print(f"正在处理 JSON 文件: {json_filepath}")
    print(f"JSON 文件目录: {json_dir}")

    errors = [] # 收集处理图片时遇到的错误

    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        msg = f"错误：process_json_with_images: 解码 JSON 失败: {e}"
        print(msg)
        errors.append(msg)
        return None, errors
    except Exception as e:
        msg = f"错误：process_json_with_images: 读取 JSON 文件失败: {e}"
        print(msg)
        errors.append(msg)
        return None, errors

    if not isinstance(data, list):
        msg = "警告：process_json_with_images: JSON 数据不是列表。脚本预期顶层 JSON 是一个块列表。"
        print(msg)
        errors.append(msg)
        return None, errors

    blocks_list = data
    modified_count = 0

    for i, block in enumerate(blocks_list):
        if isinstance(block, dict) and block.get("type") == "image":
            img_relative_path = block.get("img_path")

            if img_relative_path:
                img_full_path = os.path.join(json_dir, img_relative_path)

                print(f"\n发现图片块（索引 {i}）。正在处理图片: {img_full_path}")

                if not os.path.exists(img_full_path):
                    msg = f"警告：图片文件未找到于 {img_full_path}。跳过此块处理。"
                    print(msg)
                    block["img_text"] = msg # 在JSON中记录错误
                    errors.append(msg)
                    modified_count += 1
                    continue

                try:
                    prompt = "请识别出图中所有的文字"
                    response = inference_with_api(img_full_path, prompt)

                    print(f"图片（索引 {i}）的文本结果:\n---\n{response}\n---")

                    block["img_text"] = response
                    modified_count += 1
                    print(f"已为图片块（索引 {i}）添加 'img_text'。")

                except Exception as e:
                    msg = f"错误：处理图片 {img_full_path}（索引 {i}）时出错: {e}"
                    print(msg)
                    block["img_text"] = msg # 在JSON中记录错误
                    errors.append(msg)
                    modified_count += 1

    base, ext = os.path.splitext(json_filepath)
    output_filepath = f"{base}_processed{ext}"

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\n处理完成。共处理了 {modified_count} 个图片块。")
        if errors:
            print(f"警告：处理过程中遇到 {len(errors)} 个图片处理错误。")
        print(f"修改后的 JSON 已保存至: {output_filepath}")
        return output_filepath, errors # 返回生成的处理后 JSON 文件路径和错误列表
    except Exception as e:
        msg = f"错误：保存修改后的 JSON 文件失败: {e}"
        print(msg)
        errors.append(msg)
        return None, errors # 返回 None 表示保存失败，并附带错误列表


# --- 您的 TXT 转 JSON 函数 ---
# 保持与您提供的代码一致，并调整以接受 output_path 参数
def split_text_by_punctuation(text, max_chunk_length=200):
    # ... (函数体与您提供的代码相同) ...
    sentences = re.split(r'(?<=[，。,.])', text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_chunk_length:
            chunk += sentence
        else:
            if chunk.strip():
                chunks.append(chunk.strip())
            chunk = sentence
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

def split_and_save_to_json(file_path: str, output_path: str, max_chunk_length: int = 200) -> str | None:
    # Modified return type
    """
    Reads a text file, splits it into chunks by punctuation, and saves to a JSON file.

    Args:
        file_path (str): Path to the input text file.
        output_path (str): Path where the output JSON file will be saved.
        max_chunk_length (int): Maximum length for text chunks.

    Returns:
        str | None: The path to the generated JSON file, or None if an error occurred.
    """
    if not os.path.exists(file_path):
        print(f"错误：split_and_save_to_json: 输入文件未找到于 {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"错误：split_and_save_to_json: 读取文件失败 {file_path}: {e}")
        return None

    paragraphs = text.split('\n\n')
    all_chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        chunks = split_text_by_punctuation(para, max_chunk_length)
        all_chunks.extend(chunks)

    json_data = [{"type": "txt", "text": chunk} for chunk in all_chunks]

    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"✅ JSON 文件已保存到: {output_path}")
        return output_path
    except Exception as e:
        print(f"错误：split_and_save_to_json: 保存 JSON 文件失败 {output_path}: {e}")
        return None


# --- 您的 Markdown 转 JSON 函数 ---
# 保持与您提供的代码一致，并调整以接受 output_path 参数
def markdown_to_json(file_path: str, output_path: str) -> str | None:
    # Modified return type
    """
    Reads a Markdown file, converts relevant parts to HTML, extracts text from
    headers and paragraphs, and saves to a JSON file.

    Args:
        file_path (str): Path to the input Markdown file.
        output_path (str): Path where the output JSON file will be saved.

    Returns:
        str | None: The path to the generated JSON file, or None if an error occurred.
    """
    if not os.path.exists(file_path):
        print(f"错误：markdown_to_json: 输入文件未找到于 {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()
    except Exception as e:
        print(f"错误：markdown_to_json: 读取文件失败 {file_path}: {e}")
        return None

    html_content = markdown.markdown(markdown_content)

    soup = BeautifulSoup(html_content, "html.parser")

    result = []

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        block = {}
        if tag.name.startswith('h'):
             block["type"] = f"heading_{tag.name[1]}"
        elif tag.name == "p":
             block["type"] = "paragraph"

        text_content = tag.get_text().strip()
        if not text_content:
            continue

        block["text"] = text_content
        result.append(block)

    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
        print(f"JSON 文件已保存到: {output_path}")
        return output_path
    except Exception as e:
        print(f"错误：markdown_to_json: 保存 JSON 文件失败 {output_path}: {e}")
        return None

# --- 新增：解压缩函数 ---
def extract_zip_archive(archive_path: str, extract_to_dir: str) -> str | None:
    """
    解压缩 zip 文件到指定目录。

    Args:
        archive_path (str): zip 文件的完整路径。
        extract_to_dir (str): 解压目标目录。

    Returns:
        str | None: 解压成功后，解压目标目录的路径；如果出错，返回 None。
    """
    if not os.path.exists(archive_path):
        print(f"错误：extract_zip_archive: 压缩文件未找到于 {archive_path}")
        return None

    try:
        os.makedirs(extract_to_dir, exist_ok=True)
        print(f"正在解压 {archive_path} 到 {extract_to_dir}")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
        print("解压完成。")
        return extract_to_dir
    except zipfile.BadZipFile:
        print(f"错误：extract_zip_archive: 文件 {archive_path} 不是有效的 Zip 文件。")
        return None
    except Exception as e:
        print(f"错误：extract_zip_archive: 解压文件失败 {archive_path}: {e}")
        return None

# --- 新增：处理单个文件的内部函数 ---
# 这是一个内部函数，封装了之前在 API 端点中的单个文件处理逻辑
# 这样就可以在批量处理时复用
def _process_single_file(file_path: str, output_base_dir: str) -> 'SingleFileProcessResult':
     """
     处理一个单个文件 (TXT, MD, 或 magic-pdf 支持的类型)。

     Args:
         file_path (str): 输入文件的完整路径。
         output_base_dir (str): 处理结果输出文件的**根目录**。

     Returns:
         SingleFileProcessResult: 包含处理结果的 Pydantic 模型实例。
     """
     filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
     file_ext = os.path.splitext(file_path)[1].lower()

     processed_json_path = None
     magic_pdf_stdout = None
     magic_pdf_stderr = None
     magic_pdf_return_code = None
     image_processing_errors: List[str] = []
     status = "error" # Default status
     message = f"未知错误或文件类型 {file_ext} 不支持" # Default message


     print(f"\n--- Processing single file: {file_path} (Type: {file_ext}) ---")

     # 确保输出根目录存在（外层已检查，这里作为二次确认）
     try:
         os.makedirs(output_base_dir, exist_ok=True)
     except Exception as e:
         message = f"错误：创建或访问输出根目录失败 {output_base_dir}: {e}"
         print(message)
         return SingleFileProcessResult(
             original_file_path=file_path,
             status=status,
             message=message
         )


     if file_ext == ".txt":
         print(f"--- 进行 TXT 转 JSON 处理 ---")
         output_json_path = os.path.join(output_base_dir, f"{filename_without_ext}.json")
         try:
             generated_path = split_and_save_to_json(file_path, output_json_path)
             if generated_path:
                 status = "success"
                 message = "TXT 文件成功转换为 JSON"
                 processed_json_path = generated_path
             else:
                 message = f"TXT 转 JSON 失败" # split_and_save_to_json 会打印详细错误

         except Exception as e:
              status = "error"
              message = f"TXT 转 JSON 过程中发生错误: {e}"
              print(message)


     elif file_ext == ".md":
         print(f"--- 进行 Markdown 转 JSON 处理 ---")
         output_json_path = os.path.join(output_base_dir, f"{filename_without_ext}.json")
         try:
             generated_path = markdown_to_json(file_path, output_json_path)
             if generated_path:
                 status = "success"
                 message = "Markdown 文件成功转换为 JSON"
                 processed_json_path = generated_path
             else:
                 message = f"Markdown 转 JSON 失败" # markdown_to_json 会打印详细错误
         except Exception as e:
              status = "error"
              message = f"Markdown 转 JSON 过程中发生错误: {e}"
              print(message)

     elif file_ext in [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]: # Add other magic-pdf supported extensions here
         # Default case: Assume magic-pdf can handle it
         print(f"--- 使用 magic-pdf 进行解析 ---")

         # 检查 magic-pdf 路径是否存在
         if not os.path.exists(MAGIC_PDF_PATH):
             status = "error"
             message = f"未配置 magic-pdf 可执行文件路径或文件不存在于 {MAGIC_PDF_PATH}。"
             print(message)
         else:
             # magic-pdf 将在 output_base_dir 下创建子文件夹结构 (e.g., filename/auto/)
             return_code, stdout, stderr = run_magic_pdf(file_path, output_base_dir)

             magic_pdf_return_code = return_code
             magic_pdf_stdout = stdout
             magic_pdf_stderr = stderr

             if return_code != 0:
                 status = "error"
                 message = "magic-pdf 解析失败"
                 print(message)
             else:
                 print("\n--- magic-pdf 解析成功 ---")

                 # --- 处理生成的 JSON 文件 (抽取图片文本) ---
                 # 根据 magic-pdf 的输出结构，构建生成的 JSON 文件路径
                 generated_json_file_path = os.path.join(
                     output_base_dir,
                     filename_without_ext,
                     "auto",
                     f"{filename_without_ext}_content_list.json"
                 )

                 print(f"\n--- 调用阶段 2: 处理 magic-pdf 生成的 JSON 文件 ---")
                 print(f"将要处理的 JSON 文件: {generated_json_file_path}")

                 # 检查生成的 JSON 文件是否存在
                 if not os.path.exists(generated_json_file_path):
                     status = "error"
                     message = f"错误：预期的 JSON 文件未找到于 {generated_json_file_path}，magic-pdf 可能执行失败或生成了非预期的文件结构。"
                     print(message)
                 else:
                     # 调用 JSON 处理函数 (会生成 *_processed.json 文件)
                     processed_path, img_errors = process_json_with_images(generated_json_file_path)
                     image_processing_errors = img_errors # Collect image processing errors

                     if processed_path:
                         status = "success"
                         message = "PDF 解析和图片文本抽取处理完成"
                         processed_json_path = processed_path
                     else:
                         status = "error"
                         message = f"处理生成的 JSON 文件失败于 {generated_json_file_path}。"
                         print(message)


     else:
         # 不支持的文件类型
         status = "error"
         message = f"不支持的文件类型: {file_ext}"
         print(message)


     # 返回处理结果
     return SingleFileProcessResult(
         original_file_path=file_path,
         status=status,
         message=message,
         processed_json_path=processed_json_path,
         magic_pdf_stdout=magic_pdf_stdout,
         magic_pdf_stderr=magic_pdf_stderr,
         magic_pdf_return_code=magic_pdf_return_code,
         image_processing_errors=image_processing_errors
     )


# --- API 端点定义 ---

# 定义请求体的数据模型 (使用 Pydantic)
class ProcessDocumentRequest(BaseModel):
    input_path: str # 输入文件或压缩包的完整路径
    output_base_dir: str # 处理结果输出文件的**根目录**

# 定义单个文件处理结果的数据模型
class SingleFileProcessResult(BaseModel):
    original_file_path: str # 原始文件路径 (解压后的内部路径)
    status: str # "success", "error", or "skipped" (if file type is ignored)
    message: str # 描述信息
    processed_json_path: Optional[str] = None # 生成的处理后 JSON 文件路径
    # Fields specific to magic-pdf process (optional)
    magic_pdf_stdout: Optional[str] = None
    magic_pdf_stderr: Optional[str] = None
    magic_pdf_return_code: Optional[int] = None
    # Errors specific to image processing (optional, only relevant for magic-pdf output)
    image_processing_errors: Optional[List[str]] = None

# 定义 API 响应的数据模型 (现在可以包含单个或多个文件的处理结果)
class ProcessDocumentResponse(BaseModel):
    status: str # "success" or "error" for the overall request
    message: str # Overall description
    # For single file input, processed_json_path will be here
    processed_json_path: Optional[str] = None
    # For archive input, results_list will contain results for each file
    results_list: Optional[List[SingleFileProcessResult]] = None
    # Add archive specific info if needed, e.g.,
    # archive_path: Optional[str] = None
    # extracted_to_dir: Optional[str] = None
    # cleanup_successful: Optional[bool] = None


@app.post("/process_document", response_model=ProcessDocumentResponse)
async def process_document(request: ProcessDocumentRequest):
    """
    通过API触发文档处理流程，支持单个文件 (PDF, TXT, Markdown 等) 或 ZIP 压缩包。

    - 如果输入是 ZIP 压缩包，解压后处理其中所有支持的文件。
    - 对于 TXT 和 Markdown 文件，直接转换为 JSON 格式。
    - 对于其他文件格式 (如 PDF, DOCX等)，使用 magic-pdf 进行解析，然后处理生成的 JSON 抽取图片文本。
    """
    input_path = request.input_path
    output_base_dir = request.output_base_dir

    print(f"\n--- API Request: Processing input {input_path} into {output_base_dir} ---")

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
         raise HTTPException(status_code=400, detail={"status": "error", "message": f"输入文件未找到于 {input_path}"})

    # 检查输出目录是否存在，并尝试创建
    try:
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"确保输出根目录存在: {output_base_dir}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"创建或访问输出根目录失败 {output_base_dir}: {e}"}
        )

    file_ext = os.path.splitext(input_path)[1].lower()
    filename_without_ext = os.path.splitext(os.path.basename(input_path))[0]

    results_list: List[SingleFileProcessResult] = [] # To collect results for batch processing

    # --- 检查是否是压缩包 ---
    if file_ext == ".zip":
        print(f"--- 检测到 ZIP 压缩包 ({input_path})，进行解压和批量处理 ---")
        # 创建一个临时目录用于解压，放在 output_base_dir 下
        extract_dir = os.path.join(output_base_dir, f"{filename_without_ext}_extracted")
        extracted_path = None
        try:
             extracted_path = extract_zip_archive(input_path, extract_dir)
             if not extracted_path:
                 raise Exception("解压失败") # Trigger the except block if extraction fails

             # 遍历解压后的目录
             processed_count = 0
             for root, _, files in os.walk(extracted_path):
                 for file in files:
                     file_path = os.path.join(root, file)
                     # 可以添加逻辑跳过不需要处理的文件类型或隐藏文件
                     if file.startswith('.') or os.path.isdir(file_path): continue # Skip hidden files and directories (though os.walk handles dirs)

                     # 对压缩包内的每个文件进行处理
                     # 将 output_base_dir 传递给 _process_single_file，让它在此目录下生成结果
                     single_file_result = _process_single_file(file_path, output_base_dir)
                     results_list.append(single_file_result)
                     if single_file_result.status == "success":
                         processed_count += 1


             # --- 清理临时解压目录 (可选) ---
             # try:
             #     print(f"\n--- 清理临时解压目录: {extracted_path} ---")
             #     shutil.rmtree(extracted_path)
             #     print("清理完成。")
             #     # response_details["cleanup_successful"] = True
             # except Exception as e:
             #     print(f"警告：清理临时解压目录失败 {extracted_path}: {e}")
             #     # response_details["cleanup_successful"] = False
             # --- 清理结束 ---


             # 汇总批量处理结果
             overall_status = "success" if processed_count > 0 and all(r.status != "error" for r in results_list) else "error"
             overall_message = f"批量处理完成。成功处理 {processed_count} 个文件，共 {len(results_list)} 个文件。"
             if overall_status == "error":
                 overall_message = f"批量处理完成，但遇到错误。成功处理 {processed_count} 个文件，共 {len(results_list)} 个文件。请查看 results_list 详情。"


             return ProcessDocumentResponse(
                 status=overall_status,
                 message=overall_message,
                 results_list=results_list # 返回所有文件的处理结果列表
             )

        except Exception as e:
             # 捕获解压或遍历文件时发生的错误
             error_msg = f"处理压缩包 {input_path} 时发生错误: {e}"
             print(error_msg)
             # 在失败时也返回部分已处理的结果列表，以及总体错误信息
             return ProcessDocumentResponse(
                 status="error",
                 message=error_msg,
                 results_list=results_list # 返回部分或全部文件的处理结果
             )
        finally:
             # 无论成功失败，如果在提取步骤成功创建了目录，尝试清理（即使处理文件时失败了）
             # 如果不需要清理，可以移除此 finally 块或其中的 rmtree 调用
             if extracted_path and os.path.exists(extracted_path):
                 try:
                    print(f"\n--- 尝试清理临时解压目录: {extracted_path} ---")
                    shutil.rmtree(extracted_path)
                    print("清理完成。")
                 except Exception as e:
                    print(f"警告：清理临时解压目录失败 {extracted_path}: {e}")


    else:
        # --- 不是压缩包，按单个文件处理 (现有逻辑) ---
        print(f"--- 检测到单个文件 ({input_path})，按单个文件处理 ---")
        # 调用新的内部处理函数来处理单个文件
        single_file_result = _process_single_file(input_path, output_base_dir)

        # 根据单个文件的结果构建最终响应
        if single_file_result.status == "success":
             return ProcessDocumentResponse(
                 status="success",
                 message="单个文件处理完成",
                 processed_json_path=single_file_result.processed_json_path, # 返回主结果路径
                 # 将 magic-pdf 和 image processing 的详细信息也包含在顶层响应中
                 magic_pdf_stdout=single_file_result.magic_pdf_stdout,
                 magic_pdf_stderr=single_file_result.magic_pdf_stderr,
                 magic_pdf_return_code=single_file_result.magic_pdf_return_code,
                 image_processing_errors=single_file_result.image_processing_errors,
                 results_list=None # 单文件处理时 results_list 为 None
             )
        else:
             # 如果单个文件处理失败，返回错误响应
             raise HTTPException(
                 status_code=500, # 假设是服务器处理内部错误
                 detail={
                     "status": "error",
                     "message": f"单个文件处理失败: {single_file_result.message}",
                     "original_file_path": single_file_result.original_file_path,
                     "processed_json_path": single_file_result.processed_json_path, # 可能为 None
                     "magic_pdf_stdout": single_file_result.magic_pdf_stdout,
                     "magic_pdf_stderr": single_file_result.magic_pdf_stderr,
                     "magic_pdf_return_code": single_file_result.magic_pdf_return_code,
                     "image_processing_errors": single_file_result.image_processing_errors,
                 }
             )


# 健康检查端点
@app.get("/health")
async def health_check():
    # ... (函数体与之前的代码相同) ...
    """
    API 健康检查。检查服务是否运行，并可选地检查 magic-pdf 可执行文件是否存在。
    """
    health_status = {"status": "ok", "message": "Service is running."}
    if not os.path.exists(MAGIC_PDF_PATH):
         health_status["status"] = "warning"
         health_status["message"] += f" WARNING: magic-pdf executable not found at {MAGIC_PDF_PATH}. PDF processing might fail."
    return health_status

# 您可以在这里添加其他辅助函数或端点