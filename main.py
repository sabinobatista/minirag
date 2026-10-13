from sentence_transformers import SentenceTransformer
from pathlib import Path
import pymupdf as pymu 
import numpy as np
import json


PASTA_DADOS = Path(__file__).resolve().parent / "data"
PASTA_PROJETO = Path(__file__).resolve().parent


def carregar_pdfs():
    documentos = []

    for arquivo in PASTA_DADOS.glob("*.pdf"):
        with pymu.open(arquivo) as pdf:
            for numero_pagina, pagina in enumerate(pdf, start=1):
                texto = pagina.get_text("text").strip()

                if texto:
                    documentos.append({
                        "arquivo": arquivo.name,
                        "pagina": numero_pagina,
                        "texto": texto
                    })

    return documentos


def dividir_texto(texto, tamanho=150, sobreposicao=30):
    palavras = texto.split()
    chunks = []

    passo = tamanho - sobreposicao

    for inicio in range(0, len(palavras), passo):
        trecho = palavras[inicio:inicio + tamanho]

        if trecho:
            chunks.append(" ".join(trecho))

    return chunks



if __name__ == "__main__":
    documentos = carregar_pdfs()

    print(f"Total de páginas com texto: {len(documentos)}")

    if documentos:

        chunks = []

        for documento in documentos:
            textos_divididos = dividir_texto(documento["texto"])

            for texto in textos_divididos:
                chunks.append({
                    "arquivo": documento["arquivo"],
                    "pagina": documento["pagina"],
                    "texto": texto
                })

        print(f"Total de chunks: {len(chunks)}")

        modelo = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        textos = [chunk["texto"] for chunk in chunks]

        embeddings = modelo.encode(
            textos,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  
        )

        print("\n--- Validação dos embeddings ---")

        print("Quantidade de chunks:", len(chunks))
        print("Quantidade de embeddings:", len(embeddings))

        print("Todos os valores são finitos:",
            np.isfinite(embeddings).all())

        print("Norma do primeiro embedding:",
            np.linalg.norm(embeddings[0]))

        print("\nPrimeiro chunk:")
        print(chunks[0]["texto"][:200])

        print("\nPrimeiros 10 valores do embedding:")
        print(embeddings[0][:10])

        np.save(PASTA_PROJETO / "embeddings.npy", embeddings)

        with open(PASTA_PROJETO / "chunks.json", "w", encoding="utf-8") as arquivo:
            json.dump(chunks, arquivo, ensure_ascii=False, indent=2)

        print("\nEmbeddings salvos em embeddings.npy")
        print("Chunks salvos em chunks.json")

    else:
        print("Nenhum PDF com texto encontrado na pasta data/.")

