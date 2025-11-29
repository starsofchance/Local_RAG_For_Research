# Colab_Based_Multimedia_RAG_For_Researchers

# Multimodal RAG with Qwen2-VL & Colab

A state-of-the-art **Multimodal Retrieval-Augmented Generation (RAG)** system designed to run entirely within Google Colab (Pro recommended for 16GB+ VRAM).

This system goes beyond text-only RAG. It **"sees"** your documents. It extracts vector charts, raster images, and diagrams from PDFs, links them to the relevant text, and uses a Vision-Language Model (VLM) to answer complex queries requiring visual analysis.

## 🌟 Key Features

* **True Multimodal Ingestion:**
  * **Raster Images:** Extracts photos, scans, and icons.
  * **Vector Graphics:** Detects and renders "invisible" vector charts (matplotlib plots, line graphs) that standard PDF tools miss.
  * **Context-Aware Linking:** Associates images with the specific text pages they appear on.

* **State-of-the-Art Models (Quantized):**
  * **Searcher:** `Qwen/Qwen3-Embedding-0.6B`.
  * **Brain:** `Qwen/Qwen2-VL-7B-Instruct` (Vision-Language Model loaded in 4-bit for memory efficiency).

* **Advanced RAG Techniques:**
  * **Deep Search:** Bypass "Vector Shadowing" (loud papers dominating results) by retrieving top-25 or top-50 chunks.
  * **Semantic Image Filtering:** Automatically hides images if the surrounding text doesn't match the query (Score Thresholding).
  * **Hybrid Chat Modes:** Switch instantly between "Search Whole Database" and "Talk to Specific Paper".
  * **Auto-Sync:** Incremental ingestion. Drop a new PDF, run one command, and it updates the database without re-processing old files.

* **Citation Tracking:** Explicitly lists the Source File and Page Number for every piece of information used.

## 🏗️ Architecture

1. **Ingestion:** `PyMuPDF` reads PDFs -> Detects Raster & Vector art -> Clusters elements -> Renders to PNG.
2. **Chunking:** Sliding window text chunking (approx 250 tokens). Images are linked to text chunks based on Page ID.
3. **Embedding:** `Qwen3` converts text chunks to 1024d vectors. Stored in a `.pkl` dataframe.
4. **Retrieval:** `Cosine Similarity` finds relevant text chunks.
5. **Filtering:** Text chunks with low scores discard their attached images (reducing noise).
6. **Generation:** `Qwen2-VL` receives the top text context + top visual pixels to generate the answer.

## 🚀 Quick Start

### 1. Prerequisites

* Google Colab (T4 GPU works for small tests, L4/A100 recommended for production speed).
* Hugging Face Token (optional, for gated models, though Qwen is open).

### 2. Installation

Run this in the first cell of your notebook:

```python
!pip install -U torch pymupdf tqdm sentence-transformers accelerate bitsandbytes flash-attn qwen-vl-utils
````

### 3\. Initialize the System

Copy the `MultimodalRAG` class definition (see `multimodal_rag.py` or the code block below) into your notebook. Then run:

```python
# 1. Initialize (Loads database instantly)
rag = MultimodalRAG()

# 2. Load the AI Models (Takes ~1-2 mins)
rag.load_models()
```

### 4\. Ingest Data

1.  Create a folder named `PDF_Files` in Colab.
2.  Upload your research papers (`.pdf`) into that folder.
3.  Run the auto-sync command:

<!-- end list -->

```python
# Automatically finds new files, extracts images/text, and updates the DB
rag.ingest_new_files()
```

### 5\. Chat

You are ready to query your knowledge base.

```python
# Mode 1: Search the whole database
rag.chat("What are the main security threats to VLMs?", top_visuals=2)

# Mode 2: Talk to a specific paper (Deep Read)
rag.chat("Explain the methodology in Figure 2", 
         target_file="Prompt_Injection_Attacks.pdf", 
         top_visuals=4)
```

## 🛠️ Configuration & Controls

The `chat()` function gives you granular control over the pipeline:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `target_file` | `None` | If set (e.g. `"Paper.pdf"`), context is restricted to ONLY that file. If `None`, searches the entire DB. |
| `top_k` | `25` | How many text chunks to retrieve. Increase to `50` if a specific paper is being "shadowed" by others. |
| `top_visuals` | `2` | Max number of images to feed the Vision Model. Higher = better analysis, but uses more GPU memory. |
| `score_threshold` | `0.35` | Minimum similarity score required to show an image. Prevents the model from seeing irrelevant diagrams. |

## 📂 Project Structure

```
/content/
├── PDF_Files/                  # Upload your PDFs here
│   ├── Paper_A.pdf
│   └── Paper_B.pdf
│
├── Extracted_Images/           # System auto-saves charts here
│   ├── Paper_A_p3_fig0.png
│   └── ...
│
├── multimodal_rag_embeddings_qwen.pkl  # The Vector Database (Save this!)
└── multimodal_rag.ipynb        # The main notebook
```

## 🧠 Technical Details

  * **Quantization:** `BitsAndBytes` (NF4) is used to load the 7B parameter model into \~6GB of VRAM.
  * **Vector Extraction:** Custom logic (`cluster_elements`) identifies bounding boxes of vector drawing instructions (paths, lines) and renders them as bitmaps, solving the "missing charts" problem common in PDF parsers.
  * **Memory Management:** The system is designed to run `ingest` (CPU/RAM heavy) and `chat` (GPU VRAM heavy) sequentially.

## 🤝 Contributing

Feel free to open issues if you encounter PDFs with complex layouts that the extraction script misses.

## 📜 License

MIT License. Free to use for research and commercial applications.

```
```
