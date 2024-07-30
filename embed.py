
# encoding = "utf-8"
 
# # 加载txt文本文件
# from langchain_community.document_loaders import TextLoader
# loader = TextLoader(file_path="./data/jb1.txt", encoding="utf-8")
# doc_txt = loader.load()
# print(doc_txt)
 
# # 加载csv文件
# from langchain_community.document_loaders import CSVLoader
# loader = CSVLoader(file_path="xxx.csv", encoding="utf-8")
# doc_csv = loader.load()
# print(doc_csv)
 
# # 加载目录中的所有文件
# from langchain_community.document_loaders import DirectoryLoader
# loader = DirectoryLoader(path='xxx')
# doc_dir = loader.load()
 
# # 加载网页文件
# from langchain_community.document_loaders import BSHTMLLoader
# loader = BSHTMLLoader(file_path="xxx.html")
# doc_html = loader.load()
 
# # 加载JSON格式文件
# from langchain_community.document_loaders import JSONLoader
# loader = JSONLoader(file_path="xxx.json", jq_schema='.[]')
# doc_json = loader.load()
 
# # 加载markdown文件
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
# loader = UnstructuredMarkdownLoader(file_path="xxx.json", mode="elements")
# doc_md = loader.load()
 
# 使用 pyPDF 文档加载器，将PDF文档加载为文档数组，数组中的每个文档包含页面内容和页码的元数据
# from langchain_community.document_loaders import MathpixPDFLoader
# loader = MathpixPDFLoader("THE+PROMPT+ENGINEERING+GUIDE.pdf")
# data = loader.load()
# # # 也可以使用 UnstructuredPDFLoader 加载, 为不同的文本块创建不同的元素，默认情况下，它会将这些元素合并在一起，通过mode="elements"来分离这些元素
# from langchain_community.document_loaders import UnstructuredPDFLoader
# loader = UnstructuredPDFLoader("xxx.pdf")
 
# # 在线 PDF 文档加载
# from langchain_community.document_loaders import OnlinePDFLoader
# loader = OnlinePDFLoader("http://xxxx.pdf")
 
# # PyPDFium2 文档加载器
from langchain_community.document_loaders import PyPDFium2Loader
loader = PyPDFium2Loader("THE+PROMPT+ENGINEERING+GUIDE.pdf")
data = loader.load()

# # PDFMiner 文档加载器，可生成 HTML 文档, 进而通过 Python 的 BeautifulSoup 库进行解析和处理
# from langchain_community.document_loaders import PDFMinerLoader
# loader = PDFMinerLoader("THE+PROMPT+ENGINEERING+GUIDE.pdf")
 
# # PyMuPDF 文档加载器，最快的PDF文档加载器，输出的文档包含关于 PDF 及其页面的详细元数据，且为每页返回一个文档
# from langchain_community.document_loaders import PyMuPDFLoader
# loader = PyMuPDFLoader("THE+PROMPT+ENGINEERING+GUIDE.pdf")
 
# # PyPDFDirectoryLoader 文档加载器，可从目录加载 PDF 文档
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# loader = PyPDFDirectoryLoader("THE+PROMPT+ENGINEERING+GUIDE.pdf")
 
# # PDFPlumberLoader 文档加载器，与PyMuPDF文档加载器类似，输出的文档包含关于 PDF 及其页面的详细元数据，且为每页返回一个文档。
# from langchain_community.document_loaders import PDFPlumberLoader
# loader = PDFPlumberLoader("THE+PROMPT+ENGINEERING+GUIDE.pdf")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import chroma

# 示例文档
# raw_documents = [
#     "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
#     "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
#     "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
#     "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
#     "Llamas are vegetarians and have very efficient digestive systems",
#     "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
# ]

# # 使用LangChain进行文本分割
# documents = [{"content": doc, "metadata": {}} for doc in raw_documents]

# # 定义一个辅助函数来转换为 LangChain 的 Document 对象
# class Document:
#     def __init__(self, page_content, metadata=None):
#         self.page_content = page_content
#         self.metadata = metadata if metadata else {}

# # 创建 Document 对象列表
# docs = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in documents]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
# for split in all_splits:
#     print(split.page_content)

# # 设置嵌入模型
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text:v1.5")

# # 创建Chroma向量数据库，并添加分割后的文档
vectorstore = chroma.Chroma.from_documents(documents=all_splits, embedding=oembed)

# question = "What about Prompt Formula"
# response = oembed.embed_query(question)
# results = vectorstore.similarity_search(question)

# # 打印查询结果
# for result in results:
#     print(result)


from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
llm = Ollama(base_url='http://localhost:11434', model='mymodel')

prompt_template = PromptTemplate(
    input_variables=["context","question"],
    template="""
    Use the following pieces of context to answer the question.
    Context:{context}
    Question:{question}
    """
)
llm_chain = LLMChain(llm=llm,prompt=prompt_template)

from langchain.chains.combine_documents.stuff import StuffDocumentsChain 
combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name='context'
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=combine_documents_chain
)


def answer_uniswap_question(question):
  result = qa_chain.run(question)
  return result

config_list = [
    {
        "model": "LMstudio/llama3-zh",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio"
    }
]

llm_config={
    "config_list": config_list,
    "temperature": 0,
    "functions": [
        {
            "name": "answer_uniswap_question",
            "description": "Answer any prompt related questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask in relation to prompt",
                    }
                },
                "required": ["question"],
            },
        },
    ],
}
# import autogen
# assistant = autogen.AssistantAgent(
#     name="assistant",
#     llm_config=llm_config,
#     system_message="You are an expert in prompt engineering with years of experience optimizing various AI model prompts to improve and enhance the prompts to improve the performance of AI models."
# )
# # create a UserProxyAgent instance named "user_proxy"
# user_proxy = autogen.UserProxyAgent(
#     name="user_proxy",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=10,
#     code_execution_config={"use_docker": False },
#     llm_config=llm_config,
#     system_message="You're an experienced AI QA expert with a focus on evaluating the effectiveness of prompts, a keen eye for detail, and a deep understanding of user needs.",
#     function_map={"answer_uniswap_question": answer_uniswap_question}
# )



"""
    主函数，处理用户输入并启动优化流程
"""
print("Welcome to the Prompt Optimization Assistant!")

# 获取用户输入
original_prompt = input("Please enter the prompt you want to optimize: ")
task_type = input("What type of task is this prompt for? ")
user_requirements = input("Please enter any specific requirements for the optimized prompt: ")

print(f"\nThank you! Starting the optimization process for a {task_type} prompt with the following requirements: {user_requirements}\n")

from autogen import ConversableAgent

jack = ConversableAgent(
    "Jack",
    llm_config=llm_config,
    #system_message="Your name is Jack and you are a comedian in a two-person comedy show.",
    system_message="Your name is Jack, and you are a prompt engineering expert with years of experience optimizing various AI model prompts to improve and enhance prompts to improve AI model performance. You will optimize original_prompt based on the user's task_type and userr_requirements, and modify your optimization results based on Emma's analysis results and recommendations. Every time you receive Emma's advice, give the complete optimization results with revisions and improvements based on Emma's suggestions.",
    function_map={"answer_uniswap_question": answer_uniswap_question},
)
emma = ConversableAgent(
    "Emma",
    llm_config=llm_config,
    #system_message="Your name is Emma and you are a comedian in two-person comedy show.",
    system_message="Your name is Emma, and you're an experienced AI QA expert with a focus on evaluating the effectiveness of prompts, a keen eye for detail, and a deep understanding of user needs. You will analyze Jack's optimization results and suggest your improvements, which will be analyzed and suggested for each round of dialogue.",
    function_map={"answer_uniswap_question": answer_uniswap_question},
)

chat_result = emma.initiate_chat(jack, message=f'''Jack,please help me optimize the following prompt:
            {original_prompt}.The task_type is {task_type} and the user_requirements is {user_requirements}. 
            Please output the improved results.''',
            max_turns=3)

