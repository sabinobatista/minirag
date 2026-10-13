# MiniRAG — busca semântica e respostas com LLM local

MiniRAG é um projeto de estudo que implementa um fluxo simples de **Retrieval-Augmented Generation (RAG)** usando um paper sobre Word2Vec como documento de referência.

O projeto extrai texto de um PDF, divide o conteúdo em chunks, gera embeddings, recupera os trechos mais relevantes para uma pergunta e utiliza um modelo de linguagem local para formular uma resposta com base nesses trechos.

## Tecnologias

- **Python** — implementação do pipeline.
- **PyMuPDF** — extração de texto do PDF.
- **Sentence Transformers (`all-MiniLM-L6-v2`)** — geração dos embeddings.
- **NumPy** — armazenamento e comparação dos vetores.
- **Ollama** — execução local do LLM.
- **Qwen2.5 7B** — modelo de linguagem usado por padrão.

## Estrutura do projeto

```text
minirag/
├── data/
│   └── 1301.3781v3.pdf
├── main.py
├── busca.py
├── mini_rag.py
├── requirements.txt
├── README.md
├── .gitignore
├── chunks.json
└── embeddings.npy
```

`chunks.json` e `embeddings.npy` são os artefatos gerados pelo processamento do documento. Eles podem ser recriados executando `main.py`.

## Requisitos

- Python 3.10 ou superior recomendado.
- Ollama instalado para executar o MiniRAG completo.
- Espaço em disco para as dependências Python, o modelo de embeddings e o modelo LLM.
- Conexão com a internet na primeira instalação para baixar dependências e modelos.

## Instalação

Abra o terminal na pasta raiz do projeto.

### 1. Criar e ativar um ambiente virtual

No Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se a ativação estiver bloqueada pela política do PowerShell, consulte a documentação do Python ou execute os comandos usando o interpretador do ambiente virtual diretamente.

### 2. Instalar as dependências

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Instalar o Ollama e baixar o modelo

Instale o Ollama pelo site oficial: <https://ollama.com/download>

Depois, no terminal:

```powershell
ollama pull qwen2.5:7b
```

Verifique se o modelo responde:

```powershell
ollama run qwen2.5:7b
```

Digite uma pergunta de teste e use `/bye` para sair.

## Como executar

Execute os comandos a partir da pasta raiz do projeto, onde estão `main.py`, `busca.py` e `mini_rag.py`.

### 1. Processar o PDF e gerar os embeddings

```powershell
python main.py
```

O script procura PDFs em `data/`, extrai o texto, divide cada página em chunks com sobreposição, gera embeddings e grava:

- `chunks.json`
- `embeddings.npy`

A execução novamente substitui esses arquivos pelos artefatos gerados a partir dos PDFs encontrados.

### 2. Fazer uma busca semântica

```powershell
python busca.py
```

Digite uma pergunta quando solicitado. O programa mostra os chunks com maior similaridade de cosseno, junto das páginas de origem e dos scores.

### 3. Fazer uma pergunta ao MiniRAG completo

Com o Ollama em execução e o modelo baixado:

```powershell
python mini_rag.py
```

O programa recupera os cinco chunks mais relevantes por padrão e envia a pergunta e os trechos ao LLM para gerar uma resposta.

## Como funciona

1. **Extração:** o texto é extraído do PDF com PyMuPDF.
2. **Chunking:** o texto é dividido em trechos de aproximadamente 150 palavras, com sobreposição de 30 palavras.
3. **Embeddings:** cada chunk é convertido em um vetor usando `sentence-transformers/all-MiniLM-L6-v2`.
4. **Recuperação:** a pergunta também é convertida em embedding e comparada com os vetores armazenados por produto escalar entre vetores normalizados, equivalente à similaridade de cosseno.
5. **Geração:** o Ollama executa o Qwen2.5 7B para formular uma resposta usando os trechos recuperados.

## Limitações conhecidas

- É um projeto experimental e educacional, não um sistema de produção.
- A qualidade da resposta depende do texto extraído, da divisão em chunks e dos resultados da busca semântica.
- O score de similaridade não representa uma probabilidade de acerto.
- O LLM pode interpretar incorretamente os trechos ou produzir afirmações sem suporte; as respostas e referências de página devem ser verificadas.
- A busca e o carregamento dos arquivos pressupõem que os comandos sejam executados na pasta raiz do projeto.
- O modelo `all-MiniLM-L6-v2` é utilizado tanto para os embeddings dos chunks quanto para os embeddings das perguntas. Se o modelo for alterado, os embeddings devem ser regenerados.

## Documento utilizado

O documento de demonstração é o paper **“Efficient Estimation of Word Representations in Vector Space”**, de Tomas Mikolov e colaboradores.

Versão no arXiv: <https://arxiv.org/abs/1301.3781>

Consulte a página do paper para obter informações de autoria, publicação e licença antes de redistribuir o PDF no repositório.

## Próximos passos possíveis

- Melhorar a robustez dos caminhos dos arquivos.
- Adicionar testes automatizados para chunking, carregamento e recuperação.
- Avaliar a qualidade da busca com um conjunto de perguntas e trechos esperados.
- Melhorar a apresentação das fontes e a validação das respostas.
