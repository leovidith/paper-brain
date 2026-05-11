import os
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from config import GROQ_API_BASE, LLM_MODEL_NAME
from guardrails import input_guardrail, output_guardrail
from pdf_processing import load_and_index_pdfs
from chain import create_conversational_chain


def save_chat_logs(chat_log: list, filename_base: str, log_dir: str = "chat_logs"):
    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, f"{filename_base}.json")
    md_path = os.path.join(log_dir, f"{filename_base}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chat_log, f, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Chat with Multiple PDFs\n\n")
        for turn in chat_log:
            f.write(f"**🧑 You:** {turn['user']}\n\n")
            f.write(f"**🤖 AI:** {turn['ai']}\n\n")

    print(f"\nChat logs saved to:\n - {json_path}\n - {md_path}")


def controller(pdf_paths: list) -> list:
    vectorstore = load_and_index_pdfs(pdf_paths)
    chain = create_conversational_chain(vectorstore)

    fallback_llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        base_url=GROQ_API_BASE,
        temperature=0.4
    )

    print("\nYou can now chat with the documents. Type 'exit' to quit.\n")

    chat_history = []
    chat_log = []
    eval_samples = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"chat_history_{timestamp}"

    while True:
        query = input("🧑 You: ").strip()

        if not query or query.lower() in ("exit", "quit"):
            print("Ending chat.")
            break

        allowed, sanitized_or_reason = input_guardrail(query)
        if not allowed:
            print(f"🤖 AI: {sanitized_or_reason}\n")
            continue

        result = chain.invoke({
            "question": sanitized_or_reason,
            "chat_history": chat_history
        })

        answer = result["answer"]
        sources = result.get("source_documents", [])

        if not sources:
            print("  [Fallback] No documents retrieved — using general LLM.")
            fb = fallback_llm.invoke(
                f"Answer this question as helpfully as possible: {sanitized_or_reason}"
            )
            answer = f"[Fallback — no relevant documents found]\n{fb.content}"
            sources = []

        answer, was_flagged = output_guardrail(answer, sources)

        seen = set()
        citations = "\n".join(
            f"[Source: {src} | Page: {pg}]"
            for doc in sources
            if (key := (src := doc.metadata.get("source"),
                        pg := doc.metadata.get("page", "?")))
            not in seen and not seen.add(key)
        )

        print(f"🤖 AI: {answer}\n{citations}\n")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))

        chat_log.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "user": query,
            "ai": answer,
            "sources": [{"file": d.metadata.get("source"), "page": d.metadata.get("page")} for d in sources],
            "output_flagged": was_flagged,
        })

        eval_samples.append({
            "question": query,
            "answer": answer,
            "contexts": [d.page_content for d in sources],
            "ground_truth": "",
        })

    save_chat_logs(chat_log, filename_base)
    return eval_samples