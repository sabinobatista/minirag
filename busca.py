from sentence_transformers import SentenceTransformer
import numpy as np
import json

embeddings = np.load("embeddings.npy")

with open("chunks.json", "r", encoding="utf-8") as arquivo:
    chunks = json.load(arquivo)

modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def buscar(pergunta, top_k=5):

    embedding_pergunta = modelo.encode(
        pergunta,
        normalize_embeddings=True
    )

    similaridades = embeddings @ embedding_pergunta

    indices = np.argsort(similaridades)[::-1][:top_k]


    for posicao, indice in enumerate(indices, start=1):
        chunk = chunks[indice]
        score = float(similaridades[indice])

        print(f"\n{'=' * 60}")
        print(f"Resultado {posicao}")
        print(f"Similaridade: {score:.4f}")
        print(f"Arquivo: {chunk['arquivo']}")
        print(f"Página: {chunk['pagina']}")
        print(f"\n{chunk['texto']}")


if __name__ == "__main__":
    pergunta = input("Digite sua pergunta: ")
    buscar(pergunta)
