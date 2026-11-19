import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.llm.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from dotenv import load_dotenv
load_dotenv()

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=PgVector2(collection="recipes", db_url=db_url)
# )


knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(
        collection="recipes",
        db_url=db_url,
        embedder=OllamaEmbedder(model="nomic-embed-text", dimensions=768),
    ),
)
print("Loading knowledge base...")
knowledge_base.load(recreate=False)
print("Knowledge base loaded")

storage = PgAssistantStorage(table_name='pdf_assistant',db_url=db_url)

def pdf_assistant(new: bool=False, user: str='user'):
    run_id: Optional[str]=None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if(len(existing_run_ids) > 0):
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        add_references_to_prompt=True,   # replaces tool-based search
        read_chat_history=False,
        llm=Ollama(model="llama3.2"),
        debug_mode=True,                 # shows what is sent and received
)
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)
    # assistant.print_response("How do I make Pad Thai?", markdown=True, stream=False)

if __name__ == "__main__":
    typer.run(pdf_assistant)