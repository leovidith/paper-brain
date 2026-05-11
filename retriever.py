import warnings
warnings.filterwarnings("ignore")

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import FAISS
from pydantic import PrivateAttr
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class ThresholdRetriever(BaseRetriever):
    _vectorstore: FAISS = PrivateAttr()
    _threshold: float = PrivateAttr()
    _k: int = PrivateAttr()
    _rerank_top_n: int = PrivateAttr()

    def __init__(self, vectorstore, threshold=1.8, k=10, rerank_top_n=4, **kwargs):
        super().__init__(**kwargs)
        self._vectorstore = vectorstore
        self._threshold = threshold
        self._k = k
        self._rerank_top_n = rerank_top_n

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        scored = self._vectorstore.similarity_search_with_score(query, k=self._k)
        print(f"  [FAISS scores] {[(round(float(s),3), d.metadata.get('page')) for d, s in scored[:5]]}")
        candidates = [doc for doc, score in scored if score <= self._threshold]
        if not candidates:
            print(f"  [Threshold] No docs passed — returning empty, triggering fallback")
            return []
        pairs = [[query, doc.page_content] for doc in candidates]
        ce_scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        print(f"  [Reranker] {len(candidates)} candidates → top {self._rerank_top_n} selected")
        return [doc for _, doc in ranked[:self._rerank_top_n]]