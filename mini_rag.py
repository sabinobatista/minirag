
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import chat

embeddings = np.load("embeddings.npy")

with open("chunks.json", "r", encoding="utf-8") as arquivo:
    chunks = json.load(arquivo)

modelo_embedding = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

MODELO_LLM = "qwen2.5:7b"

def buscar(pergunta, top_k=5):

    vetor_pergunta = modelo_embedding.encode(
        pergunta,
        normalize_embeddings=True
    )

    scores = embeddings @ vetor_pergunta

    indices = np.argsort(scores)[::-1][:top_k]

    resultados = []

    for indice in indices:
        resultados.append({
            "chunk": chunks[indice],
            "score": float(scores[indice])
        })

    return resultados


def gerar_resposta(pergunta, resultados):

    partes_contexto = []

    for i, resultado in enumerate(resultados, start=1):
        chunk = resultado["chunk"]

        parte = (
            f"[Fonte {i} | Página {chunk['pagina']}]\n"
            f"{chunk['texto']}"
        )

        partes_contexto.append(parte)

    contexto = "\n\n".join(partes_contexto)

  
    resposta = chat(
        model=MODELO_LLM,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that answers questions "
                    "using only the supplied document excerpts. "
                    "Answer in the same language as the user's "
                    "question. Explain the answer clearly. "
                    "Cite supporting pages using [p. X]. "
                    "If the excerpts do not contain enough "
                    "information, explicitly say so. "
                    "Do not invent facts or citations. "
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{pergunta}\n\n"
                    f"Document excerpts:\n{contexto}"
                )
            }
        ],
        options={
            "temperature": 0.2,
            "num_ctx": 4096
        }
    )

    return resposta["message"]["content"]


def main():
    pergunta = input("Digite sua pergunta: ").strip()

    if not pergunta:
        print("Digite uma pergunta válida.")
        return

    resultados = buscar(pergunta)

    print("\nBuscando informações na base de dados...")


    resposta = gerar_resposta(pergunta, resultados)

    print("\n" + "=" * 60)
    print("RESPOSTA DO MINIRAG")
    print("=" * 60)
    print(resposta)

    print("\n" + "=" * 60)
    print("FONTES RECUPERADAS")
    print("=" * 60)

    for i, resultado in enumerate(resultados, start=1):
        chunk = resultado["chunk"]
        print(
            f"{i}. Página {chunk['pagina']} "
            f"| Similaridade: {resultado['score']:.4f}"
        )


if __name__ == "__main__":
    main()
