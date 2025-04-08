import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化嵌入模型
def load_embedding_model():
    embedding_model_name = "/mnt/workspace/LLaMA-Factory/bge-large-en-v1.5"
    embedding_model_kwargs = {'device': 'cuda'}
    hf = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs={'normalize_embeddings': True}
    )
    return hf

hf = load_embedding_model()
# 加载生成模型
model_name = "/mnt/workspace/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/sft/checkpoint-336"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
import os
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_and_split(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=70,
        length_function=len,
        separators=["\n\n","\n" ,  " ",    ".",    ",",  ]
    )
    texts = text_splitter.split_documents(pages)
    return texts
def initialize_database(texts, persist_directory="Rag_file/chroma_db"):
    # 清空数据库
    if os.path.exists(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=hf)
        db.delete_collection()

    # 创建新的数据库
    db = Chroma.from_documents(
        documents=texts,
        embedding=hf,
        persist_directory=persist_directory
    )
    return db
# def initialize_database(texts, persist_directory="Rag_file/chroma_db"):
#     if os.path.exists(persist_directory):
#         db = Chroma(persist_directory=persist_directory, embedding_function=hf)
#         existing_texts = db.get(include=["metadatas", "documents"])  # 获取现有文档和元数据
#         # 将现有文档重新构造为 Document 对象
#         existing_docs = [
#             Document(page_content=content, metadata=metadata)
#             for content, metadata in zip(existing_texts["documents"], existing_texts["metadatas"])
#         ]
#     else:
#         existing_docs = []

#     # 合并新旧文档
#     all_texts = existing_docs + texts

#     # 创建新的数据库
#     db = Chroma.from_documents(
#         documents=all_texts,
#         embedding=hf,
#         persist_directory=persist_directory
#     )
#     return db

# 主程序
path = "/mnt/workspace/LLaMA-Factory/Rag_file/assignment2.pdf"
# pdf_paths = [
#     "/mnt/workspace/LLaMA-Factory/Rag_file/assignment2.pdf",  # 原始文件
#     "/mnt/workspace/LLaMA-Factory/Rag_file/Certificate.pdf"   # 新文件
# ]
pages = load_pdf_and_split(path)
texts = load_pdf_and_split(path)
# all_texts = []
# for pdf_path in pdf_paths:
#     print(f"正在处理文件：{pdf_path}")
#     texts = load_pdf_and_split(pdf_path)
#     for text in texts:
#         text.metadata["file_name"] = os.path.basename(pdf_path)  # 添加文件名作为元数据
#     all_texts.extend(texts)

# 初始化或更新数据库
persist_directory = "Rag_file/chroma_db"
db = initialize_database(texts, persist_directory)

# 创建检索器
retriever = db.as_retriever(search_kwargs={"k": 10})


def generate_text(prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def format_response(response):
    # 移除多余内容
    response = response.split("回答：", 1)[-1].strip()

    # 格式化输出
    formatted_response = "\n".join([f"- {line.strip()}" for line in response.split("\n")])
    return formatted_response


# 定义问答函数
def answer_question(question):
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])[:1024]  # 截断上下文长度
    prompt_template = f"""
    You are a helpful assistant. Based on the following context, answer the question in English
     上下文信息：
     {context}
    问题：
    {question}
    回答：
    """
    raw_answer = generate_text(prompt_template)
    clean_answer = clean_response(raw_answer)

    return clean_answer
# def answer_question(question):
#     docs = retriever.invoke(question)
#     context = "\n".join([doc.page_content for doc in docs])[:1024]  # 截断上下文长度
#     prompt_template = f"""
#     You are a helpful assistant. Based on the following context, answer the question in English
#     上下文信息：
#     {context}
#     问题：
#     {question}
#     回答：
#     """
#     answer = generate_text(prompt_template)
#     return answer
# 示例问题
question = "what is brief report's format"
response = answer_question(question)

# 打印结果
# print("问题：", question)
print( response)



