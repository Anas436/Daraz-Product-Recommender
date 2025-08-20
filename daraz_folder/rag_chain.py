from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from daraz_folder.config import Config


class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.3)
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieve or initialize session-based chat history."""
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self, category: str = None, subcategory: str = None):
        """
        Build a RAG chain with optional metadata filtering.
        Example: build_chain(category="Mobiles") will only search mobiles.
        """
        # Metadata filter (AstraDB supports dict filters)
        metadata_filter = {}
        if category:
            metadata_filter["Category"] = category
        if subcategory:
            metadata_filter["SubCategory"] = subcategory

        # Retriever with filter + top-5 chunks
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5, "filter": metadata_filter}
        )

        # Context rewriter prompt
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Rewrite the latest user question into a standalone query "
                       "using the chat history for context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an e-commerce assistant. 
             Use ONLY the provided product context (titles, categories, prices, ratings) to answer user queries.
             Format the response strictly as a numbered list, each product on a new line, like this:
             answer in this format: 
             product name - price - rating 
             for example: 
             1. iPhone 14 Pro Max - $1099 - 4.8/5
             2. Samsung Galaxy S23 - $899 - 4.7/5
             If the answer is not available in the context, reply exactly with:
             "Sorry, I could not find relevant information."
             CONTEXT:
             {context}"""),
             MessagesPlaceholder(variable_name="chat_history"),
             ("human", "{input}")
             ])


        # History-aware retriever does:
        # rewrites the query (question) with context
        # for example, if i ask or human ""tell me about the battery life i-phone 14" then the LLM will rewrite it to
        # "What is the battery life of iPhone 14?"
        # history_aware_retriever → just rewrites
        history_aware_retriever = create_history_aware_retriever(
            self.model, retriever, context_prompt
        ) #---> [question] the newly generated query([question]) will be passed to the retriever, and the retriver will fetch some relevent chunks ([context]) from the vector store.

        # QA chain takes: 
        # 1. (retrieved chunks)---> [context] 
        # 2. user question (rewritten by history_aware_retriever) --> [question]
        # 3. ("""You are an e-commerce assistant.....) --> [System instruction/promt]
        # 4. (chat history) ---> will contain two messanges:
             #i. Human messange: original human question before rewriting 
             #ii. AI messange:  
        # It returns the final answer based on the context and question.
        question_answer_chain = create_stuff_documents_chain(
            self.model, qa_prompt
        ) # ---> AI messange 
        #At the end of this, (human message + AI messange or response) gets stored in chat_history.


        # Final retrieval-augmented chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input", #---> the user’s new message will be passed in under the key "input".
            history_messages_key="chat_history",
            output_messages_key="answer"#---> the final answer will be returned under the key "answer".
        )
