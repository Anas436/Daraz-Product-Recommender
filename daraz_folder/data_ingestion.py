from langchain_astradb import AstraDBVectorStore
#from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  # my hugging face api was failing that's why i switched to local embedding 
from daraz_folder.data_converter import DataConverter
from daraz_folder.config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataIngestor:
    def __init__(self):
        #install pip install torch transformers sentence-transformers --> for local embeding 
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="daraz_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )

    def ingest(self,load_existing=True, batch_size=200):
        if load_existing==True:
            return self.vstore
        
        docs = DataConverter("Data/Daraz_product.csv").convert()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(docs)

        for i in range(0, len(docs), batch_size):
            batch = docs[i: i + batch_size]
            #print(f"[INFO] Ingesting batch {i // batch_size + 1} / {((len(docs)-1) // batch_size)+1} ...")
            self.vstore.add_documents(batch)

        return self.vstore
    

#if __name__ == "__main__":
    #ingestor = DataIngestor()

    # for the first time: set load_existing=False
    #vstore = ingestor.ingest(load_existing=False)
