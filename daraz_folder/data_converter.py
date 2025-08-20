import pandas as pd 
from langchain_core.documents import Document

class DataConverter:
    def __init__(self, file_path:str):
        self.file_path = file_path

    def convert(self):
        df = pd.read_csv(self.file_path)[["Category", "SubCategory","Title","Original Price","Positive Seller Ratings"]]

        docs = [
            Document(
                page_content=row["Title"],
                metadata={
                    "Category": row["Category"],
                    "SubCategory": row["SubCategory"],
                    "Original Price": row["Original Price"],
                    "Positive Seller Ratings": row["Positive Seller Ratings"]
                }
            ) for _, row in df.iterrows()
        ]

        return docs