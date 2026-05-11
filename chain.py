import warnings
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import GROQ_API_BASE, LLM_MODEL_NAME
from retriever import ThresholdRetriever

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given the conversation history and a follow-up question, "
               "rewrite the follow-up into a standalone question. "
               "Return ONLY the rewritten question, nothing else."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question using "
               "ONLY the context below. If the answer is not in the context, "
               "say 'I don't know based on the provided documents.'\n\n"
               "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a query expansion assistant. The user's question failed "
               "to retrieve relevant documents from a vector store. Your job is "
               "to rewrite the question in a more detailed, contextually rich way "
               "that is more likely to match relevant document chunks. "
               "Use synonyms, add domain context, and elaborate on key terms.\n\n"
               "Recent conversation context (last 3 exchanges):\n{history}\n\n"
               "Use this context to better understand what the user is looking for. "
               "Return ONLY the expanded question, nothing else."),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_conversational_chain(vectorstore: FAISS):
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        base_url=GROQ_API_BASE,
        temperature=0.1
    )

    retriever = ThresholdRetriever(vectorstore, threshold=2.0, k=10, rerank_top_n=4)
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
    expansion_chain = QUERY_EXPANSION_PROMPT | llm | StrOutputParser()

    def get_standalone_question(inputs):
        if inputs.get("chat_history"):
            return condense_chain.invoke(inputs)
        return inputs["question"]

    def retrieve_with_expansion(inputs):
        query = inputs["standalone"]
        chat_history = inputs.get("chat_history", [])

        docs = retriever.invoke(query)

        if not docs:
            print("  [Query Expansion] First retrieval failed — expanding query...")

            recent_messages = chat_history[-6:]
            history_text = "\n".join(
                f"{'User' if i % 2 == 0 else 'AI'}: {m.content}"
                for i, m in enumerate(recent_messages)
            ) if recent_messages else "No prior conversation."

            expanded = expansion_chain.invoke({
                "question": query,
                "history": history_text
            })

            print(f"  [Query Expansion] Expanded: {expanded}")

            lenient_retriever = ThresholdRetriever(vectorstore, threshold=2.2, k=10, rerank_top_n=4)
            docs = lenient_retriever.invoke(expanded)

            if not docs:
                print("  [Query Expansion] Expanded retrieval also failed — triggering fallback.")

        return docs

    chain = (
        RunnablePassthrough.assign(
            standalone=RunnableLambda(get_standalone_question)
        )
        | RunnablePassthrough.assign(
            source_documents=RunnableLambda(retrieve_with_expansion)
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(x["source_documents"]))
        )
        | RunnablePassthrough.assign(
            answer=QA_PROMPT | llm | StrOutputParser()
        )
    )

    return chain