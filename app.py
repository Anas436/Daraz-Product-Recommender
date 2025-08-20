from flask import render_template, Flask, request, Response
from prometheus_client import Counter, generate_latest
from daraz_folder.data_ingestion import DataIngestor
from daraz_folder.rag_chain import RAGChainBuilder

from dotenv import load_dotenv
load_dotenv()

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")

def create_app():
    app = Flask(__name__)

    # ✅ Load existing vector DB, otherwise ingest data
    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain = RAGChainBuilder(vector_store).build_chain()

    @app.route("/")
    def index():
        """Landing page with chatbot UI"""
        REQUEST_COUNT.inc()
        return render_template("index.html")   # loads templates/index.html

    @app.route("/get", methods=["POST"])
    def get_response():
        """Handles AJAX requests from chatbot frontend"""
        REQUEST_COUNT.inc()

        user_input = request.form["msg"]

        # Get response from RAG chain
        response = rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "daraz-user-session"}}  # branded session id
        )["answer"]

        return response

    @app.route("/metrics")
    def metrics():
        """Expose Prometheus metrics"""
        return Response(generate_latest(), mimetype="text/plain")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)