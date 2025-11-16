try:
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print("Error: 'data.txt' not found. Please create the file.")
    exit()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever()

from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="not-needed",
    model_name="llama3.2:1b"
)

from langchain_classic.chains.conversation.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it. If the question is already standalone, "
    "just return that question."
)

from langchain_core.prompts import ChatPromptTemplate
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

history_aware_retriever = create_history_aware_retriever(
    llm=model,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


print("RAG Chatbot ready! Ask me anything. Type 'exit' to quit.\n")
while True:
    query= input("You: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Bye! See You Again!")
        break

    chat_history = memory.load_memory_variables({})["chat_history"]
    response = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    memory.save_context({"input": query}, {"answer": response["answer"]})

    print(f"Chatbot: {response['answer']}\n")
