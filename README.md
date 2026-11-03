# Contrastive Delta Vectors
### Disentangled Attribute Steering for Zero-Shot Composed Fashion Retrieval

## 1. Overview

**Contrastive Delta Vectors** is a research and development project focused on improving multimodal fashion retrieval using contrastive vector representations.

The project introduces **CADS (Contrastive Attribute-Delta Steering)**, a training-free, zero-shot framework that retrieves fashion products using a combination of reference images and natural-language modification instructions.

Unlike conventional image retrieval systems that primarily search for visually similar products, this approach allows users to specify changes in attributes such as color, sleeve length, patterns, and garment type.

For example:

- Reference image: Green, short-sleeved plain shirt.
- Modification: "Find a similar green shirt with checks and full sleeves."
- Expected result: Green checkered, full-sleeved shirts.

The system uses OpenAI CLIP embeddings, contrastive vector arithmetic, and FAISS similarity search to identify relevant products.

## 2. Problem Statement

Traditional image retrieval systems often struggle with fine-grained modifications because:

- Dominant visual features, particularly color, can overshadow structural attributes.
- Original image features may persist even when contradictory modifications are requested.
- Conversational queries may contain unnecessary words that reduce semantic alignment.

This project addresses these limitations through explicit positive attribute injection, negative attribute suppression, and natural-language normalization.

## 3. Key Features

- **Multimodal Retrieval:** Combines reference images and natural-language instructions.
- **Contrastive Delta Steering:** Uses positive and negative semantic vectors to control attribute modifications.
- **Attribute Extraction:** Identifies colors, patterns, sleeve styles, and garment categories.
- **Zero-Shot Learning:** Uses a pretrained CLIP model without task-specific retraining.
- **Vector Similarity Search:** Uses FAISS for efficient nearest-neighbor retrieval.
- **Interactive Web Interface:** Provides fashion search through a Flask application.
- **Performance Evaluation:** Includes Recall@K metrics, baseline comparisons, and ablation studies.

## 4. Technical Architecture

The system follows two major processing stages.

### A. Offline Processing

1. Collect and validate fashion product images.
2. Resize and normalize images.
3. Generate 512-dimensional image embeddings using CLIP ViT-B/32.
4. Apply L2 normalization.
5. Store embeddings in a FAISS `IndexFlatIP` index.
6. Maintain product metadata for retrieval.

### B. Online Retrieval

1. Accept a reference image and modification instruction.
2. Extract relevant attributes from the instruction.
3. Generate canonical, positive, and negative text descriptions.
4. Encode descriptions using CLIP.
5. Construct a contrastive query vector.
6. Normalize the resulting vector.
7. Perform similarity search using FAISS.
8. Return the top-ranked fashion products.

### Architecture Flow

```text
             OFFLINE PROCESSING

       Fashion Product Dataset
                  |
                  v
         Image Preprocessing
                  |
                  v
            CLIP Encoder
                  |
                  v
         Image Embeddings
                  |
                  v
          L2 Normalization
                  |
                  v
            FAISS Index
                  |
                  v
          Product Metadata


              ONLINE PROCESSING

        Reference Image + Text
                  |
                  v
          CLIP Image Encoder
                  |
                  v
          Image Embedding
                  |
        Natural Language Query
                  |
                  v
          Attribute Parser
                  |
                  v
          Attribute Extraction
                  |
          +-------+-------+
          |       |       |
          v       v       v
       Canonical Positive Negative
        Prompt   Prompt   Prompt
          |       |       |
          v       v       v
          CLIP Text Encoder
                  |
                  v
        Contrastive Delta Steering
                  |
                  v
         Final Query Embedding
                  |
                  v
            FAISS Search
                  |
                  v
          Top-K Product Results
                  |
                  v
             Flask UI
```

## 5. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Vision-Language Model | OpenAI CLIP |
| Model Architecture | ViT-B/32 |
| Deep Learning | PyTorch |
| Model Integration | Hugging Face Transformers |
| Vector Search | FAISS |
| Image Processing | Pillow |
| Numerical Computation | NumPy |
| Backend | Flask |

## 6. Repository Structure

```text
Contrastive-Delta-Vectors/
│
├── embeddings/
├── fashion-iq/
├── fashion-iq-metadata/
├── figures/
├── static/
│   └── css/
├── templates/
│
├── app.py
├── build_index.py
├── download_images.py
├── evaluate.py
├── evaluate_all.py
├── generate_paper_diagrams.py
├── generate_plots.py
│
├── evaluation_results.json
├── evaluation_results_full.json
├── RESEARCH_PAPER.md
├── requirements.txt
└── README.md
```

## 7. Installation and Setup

### Prerequisites

- Python 3.9 or later
- pip package manager
- Git
- Sufficient storage for the Fashion-IQ dataset and generated embeddings

### Step 1: Clone the Repository

```bash
git clone https://github.com/shahkarnav115-beep/Contrastive-Delta-Vectors.git
cd Contrastive-Delta-Vectors
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset and Embeddings

Prepare the Fashion-IQ dataset and generate the required image embeddings and FAISS index using the repository's preprocessing and indexing scripts.

Ensure the application can access the generated index and metadata files.

### Step 5: Run the Application

```bash
python app.py
```

Open the local Flask application in your browser at:

```text
http://127.0.0.1:5000/
```

## 8. Dataset and Evaluation

The project uses the Fashion-IQ benchmark, covering three categories:

- Dresses
- Shirts
- Tops and Tees

The research evaluation uses a validated catalog of **35,726 images** and **2,469 valid candidate-target pairs**.

Evaluation metrics include:

- Recall@10
- Recall@50
- Average Recall
- Bootstrap Confidence Intervals
- Retrieval Latency

## 9. Research Results

According to the project's research evaluation:

| Metric | Result |
|---|---:|
| Catalog Images | 35,726 |
| Valid Evaluation Pairs | 2,469 |
| Top & Tee Recall@10 | 16.35% |
| Naive CLIP Fusion Recall@10 | 14.47% |
| Relative Improvement | 13.0% |
| Average Query Latency | 19.11 ms |

The CADS framework demonstrates improved Recall@10 for multi-attribute Top & Tee retrieval compared with naive linear CLIP fusion.

These results are specific to the reported experimental setup and dataset.

## 10. Future Enhancements

- Replace static attribute mapping with lightweight language models.
- Expand support for additional garment categories.
- Improve handling of complex natural-language instructions.
- Explore adaptive weighting for different fashion attributes.
- Investigate nonlinear vector-space steering techniques.
- Improve scalability for larger product catalogs.

## 11. Research Documentation

The detailed research methodology, mathematical formulation, literature review, experimental analysis, and evaluation results are available in:

[RESEARCH_PAPER.md](RESEARCH_PAPER.md)

## 12. License and Contributions

Contributions, suggestions, and research discussions are welcome.

For improvements, submit an issue or create a pull request through GitHub.

For licensing and dataset usage, refer to the applicable repository and Fashion-IQ dataset terms.

---

**Project Repository:** [Contrastive-Delta-Vectors](https://github.com/shahkarnav115-beep/Contrastive-Delta-Vectors)

**Research Focus:** Multimodal Learning | Computer Vision | Contrastive Learning | Vector Embeddings | Zero-Shot Retrieval | Information Retrieval
