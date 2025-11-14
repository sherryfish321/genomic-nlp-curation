# 🧬 Genomic Text Curation & Topic Grouping (NLP)

**Author:** Yu-Hsuan Huang  
**Duration:** 24 hours  
**Primary Skills:** Natural Language Processing (NLP), Text Mining, Clustering, Data Structuring  
**Cost:** Zero (standard Python packages only)

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Setup & Installation](#-setup--installation)
3. [Project Structure](#-project-structure)
4. [Methods & Pipeline](#-methods--pipeline)
5. [Curation Schema](#-curation-schema)
6. [Results & Visualizations](#-results--visualizations)
7. [Limitations & Error Analysis](#-limitations--error-analysis)
8. [Next Steps](#-next-steps)

---

## Overview

This project implements an automated NLP pipeline for extracting structured genomic knowledge from unstructured scientific abstracts.   

### Key Achievements
- Processed **70 genomic abstracts** from NIAGADS Alzheimer's disease publications
- Extracted **418 gene-variant-disease triplets**
- Identified **9 distinct research topics** using UMAP + K-Means
- Quantified extraction errors for future improvement

---

## Setup & Installation

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/genomic-nlp-curation.git
cd genomic-nlp-curation

# Install dependencies
pip install -q spacy==3.7.4 pandas==2.2.2 numpy==1.26.4
pip install -q sentence-transformers umap-learn scikit-learn matplotlib seaborn

# Download spaCy English model
pip install -q https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

### Quick Start

```bash
# Run the complete pipeline
python main.py

# Or use the Jupyter notebook
jupyter notebook Genomic_Text_Curation_NLP.ipynb
```
---

## Project Structure

```
genomic-nlp-curation/
│
├── data/
│   └── texts.csv                          
│
├── outputs/
│   ├── section2_hybrid_schema.json        # Extracted entities per document
│   ├── section2_error_analysis.json       # Quantified extraction errors
│   ├── section3_assignment_relations.json # Gene-variant-disease triplets
│   ├── section4_topics.json               # Clustered documents with embeddings
│   ├── section4_cluster_keywords.json     # Top TF-IDF terms per cluster
│   └── section4_cluster_summary.json      # Human-readable cluster descriptions
│
├── figures/
│   ├── umap_clusters.png                  
│   ├── cluster_distribution.png           
│   └── entity_heatmap.png                 
│
├── genomic_text_curation_&_topic_grouping_yuhsuanhuang.ipynb       
├── genomic_text_curation_&_topic_grouping_yuhsuanhuang.py                                                  
└── README.md                              
```

---

## 🔬 Methods & Pipeline

### Architecture Overview

```
┌─────────────┐
│  Input CSVs │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  PART 1: Data Loading & Prep   │
│  - Load texts.csv               │
│  - Validate columns             │
│  - Calculate text statistics    │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  PART 2: Entity Extraction      │
│  ┌───────────────────────────┐  │
│  │ A. Rule-Based (Regex)     │  │
│  │   - Variants: rs\d{3,}    │  │
│  │   - Genes: [A-Z0-9]{3,10} │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │ B. Model-Based (spaCy)    │  │
│  │   - Diseases: NER labels  │  │
│  │   - Phenotypes: ORG/GPE   │  │
│  └───────────────────────────┘  │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  PART 3: Relation Extraction    │
│  - Cartesian product (V×G×D)    │
│  - Relation keyword matching    │
│  - Evidence span extraction     │
│  - Output: 418 triplets         │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  PART 4: Topic Modeling         │
│  - Augment text with entities   │
│  - Encode: Sentence-BERT        │
│  - Reduce: UMAP (768→2D)        │
│  - Cluster: K-Means (k=9)       │
│  - Interpret: TF-IDF keywords   │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  PART 5: Outputs & Viz          │
│  - JSON exports                 │
│  - UMAP scatter plot            │
│  - Cluster distribution         │
│  - Entity heatmap               │
└─────────────────────────────────┘
```

---

### 1. Entity Extraction 

#### A. Rule-Based Extraction (High Precision)

**Variant Detection:**
```python
RSID_PATTERN = r"\brs[0-9]{3,}\b"
# Matches: rs429358, rs7412, rs1081105
```

**Gene Symbol Detection:**
```python
GENE_PATTERN = r"\b[A-Z0-9]{3,10}\b"
# Matches: APOE, BIN1, CD33, PICALM
# Blacklist: DNA, RNA, USA, COVID, GWAS
```

#### B. Model-Based Extraction (High Recall)

**Disease/Phenotype Detection:**
- **Model**: spaCy `en_core_web_sm` v3.7.1
- **Strategy**: 
  1. Primary: Labels `DISEASE`, `DIS`, `ILLNESS`, `SYMPTOM`
  2. Fallback: `ORG`, `GPE`, `NORP` containing keywords "disease", "syndrome", "disorder"
- **Example Outputs**: "Alzheimer's disease", "late-onset AD", "cognitive decline"

---

### 2. Relation Extraction

#### Relation Keywords
```python
RELATION_KEYWORDS = [
    "associated with",
    "increases risk of", 
    "linked to",
    "related to",
    "affects",
    "causes",
    "reduces",
    "protective for"
]
```

#### Triplet Generation Logic
1. **Cartesian Product**: For each document, combine all extracted `variants × genes × diseases`
2. **Relation Inference**: Match relation keywords in text (default: "associated with")
3. **Evidence Span**: Extract the shortest sentence containing all three entities

#### Example Output
```json
{
  "text_id": "32",
  "variant": "rs429358",
  "gene": "APOE",
  "phenotype": "Alzheimer's disease",
  "relation": "increases risk of",
  "evidence_span": "The APOE ε4 allele (rs429358) increases risk of Alzheimer's disease in a dose-dependent manner."
}
```

---

### 3. Topic Modeling

#### Text Augmentation Strategy
To improve clustering, raw text is augmented with extracted entities:

```python
def build_aug_text(row):
    base = row["raw_text"]
    augmented = f"{base} VARIANTS: {variants} GENES: {genes} DISEASES: {diseases} RELATIONS: {relations}"
    return augmented
```

**Example:**
```
Original: "The APOE ε4 variant increases Alzheimer's risk..."
Augmented: "The APOE ε4 variant... VARIANTS: rs429358 GENES: APOE DISEASES: Alzheimer's disease RELATIONS: increases risk of"
```

#### Embedding Model
- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Dimensions**: 768
- **Rationale**: Balance between quality and speed

#### Dimensionality Reduction
- **Algorithm**: UMAP 
- **Parameters**:
  - `n_neighbors=15`: Preserve local structure
  - `min_dist=0.1`: Allow tight clusters
  - `metric="cosine"`: Appropriate for text embeddings
- **Output**: 2D coordinates for visualization

#### Clustering
- **Algorithm**: K-Means
- **K Selection**: Elbow method → **k=9**
- **Feature Space**: Full 768-dimensional embeddings (not 2D UMAP)

#### Topic Interpretation
- **Method**: TF-IDF on augmented text within each cluster
- **Output**: Top 10 keywords per cluster

---

## Curation Schema

### Entity Schema (`section2_hybrid_schema.json`)

```json
{
  "id": "17",
  "variants": ["rs13115400", "rs1393060", "rs316341"],
  "genes": ["002", "004", "006", "009", "047..."]
  "diseases": [],
  "raw_text": "Cerebrospinal fluid (CSF) levels of amyloid-beta 42 (Abeta42)..."
}
```

**Field Descriptions:**
| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `id` | string | Document identifier | Input CSV |
| `variants` | array | Genetic variant IDs (rsIDs) | Regex extraction |
| `genes` | array | Gene symbol candidates | Regex + blacklist |
| `diseases` | array | Disease/phenotype mentions | spaCy NER |
| `raw_text` | string | Original abstract text | Input CSV |

---

### Relation Schema (`section3_assignment_relations.json`)

```json
{
  "text_id": "61",
  "variant": "rs6701713",
  "gene": "MEASURES",
  "phenotype": "the Alzheimer Disease Genetics Consortium",
  "relation": "associated with",
  "evidence_span": "IMPORTANCE: Because APOE locus variants contribute to risk of..."
}
```

**Field Descriptions:**
| Field | Type | Description | Why It Matters |
|-------|------|-------------|----------------|
| `text_id` | string | Source document ID | Traceability to original text |
| `variant` | string | Genetic variant (rsID) | Core entity for genetic studies |
| `gene` | string | Gene symbol | Links variant to biological function |
| `phenotype` | string | Disease/trait | Clinical relevance |
| `relation` | string | Association type | Differentiates "protective" vs "risk" |
| `evidence_span` | string | Supporting sentence | Curation verification |

**Why This Schema Matters:**
1. **Structured Knowledge**: Converts prose → machine-readable facts
2. **Curation Ready**: Curators can verify each triplet independently
3. **Database Integration**: Direct mapping to knowledge graphs 
4. **Provenance**: `evidence_span` enables fact-checking

---

### Topic Schema (`section4_topics.json`)

```json
{
    "id": "33",
    "cluster": 2,
    "umap_x": 3.9729130268096924,
    "umap_y": 1.5105104446411133,
    "has_relation": true,
    "relation_count": 55,
    "relation_types": ["associated with"],
    "variants": ["rs10510109", "rs2421016", "rs4734295", "rs6982393", "rs7812465"],
    "genes": ["AND", "BACKGROUND", "CONCLUSION", "DIAGRAM..."],
    "diseases": ["International Genomics of Alzheimer's"],
    "text": "BACKGROUND: Both type 2 diabetes (T2D) and Alzheimer's disease..."
}
```

---

## Results & Visualizations

### Extraction Statistics

| Metric | Count | Rate |
|--------|-------|------|
| **Documents Processed** | 70 | 100% |
| **Documents with Variants** | 35 | 50% |
| **Documents with Genes** | 70 | 100% |
| **Documents with Diseases** | 21 | 30% |
| **Documents with Relations** | 21 | 30% |
| **Total Triplets Extracted** | 418 | avg 5.97/doc |

---

### Topic Clustering Results

#### Cluster Distribution
![Cluster Distribution](figures/cluster_distribution.png)

| Cluster | Size | Top Keywords | Research Focus |
|---------|------|--------------|----------------|
| **0** | 2 | cancer, pap, snps, prostate | Cancer genetics & SNP associations |
| **1** | 9 | csf, levels, amd, clu | Biomarker studies (CSF, blood) |
| **2** | 25 | ad, load, genes, loci | Late-onset Alzheimer's GWAS |
| **3** | 4 | hs aging, aging, abcc9 | Hippocampal sclerosis & aging |
| **4** | 4 | pd, ftd, mapt, als | Neurodegenerative diseases (PD/FTD) |
| **5** | 18 | apoe, ad, load, age | APOE-focused Alzheimer's studies |
| **6** | 1 | wmhv, white matter, fa | Brain imaging (white matter) |
| **7** | 1 | intracranial volume, pi3k | Structural brain measures |
| **8** | 6 | ad, bi, bmi, hdl | Metabolic factors in AD |

**Key Observations:**
- **Cluster 2** (Late-onset AD GWAS): Largest cluster with 25 texts 
- **Clusters 6 & 7**: Single-document clusters 
- **Clusters 2 & 5**: Together represent 43 texts (61%) 

---

#### UMAP Topic Visualization
![UMAP Clusters](figures/umap_clusters.png)

**Interpretation:**
- **Larger dots**: Documents with extracted gene-variant-disease relations 
- **Color**: Cluster assignment
- **Spatial proximity**: Semantic similarity 

**Key Observations:**
1. **Cluster 2 & 5** (35.7% + 25.7% = 61.4% of corpus): 
   - Dominant Alzheimer's GWAS themes
   - Cluster 2: Broad multi-loci studies
   - Cluster 5: APOE-centric research
   - Spatially adjacent → strong thematic overlap

2. **Cluster 4** (PD/FTD, 5.7%): Distant from AD clusters → distinct research area

3. **Clusters 6 & 7** (1.4% each): 
   - Outlier documents
   - Highly specialized or methodologically unique papers
   - May benefit from manual review for potential mis-clustering

4. **Cluster 0** (2.9%): Small cancer genetics cluster
   - Possible cross-domain papers

---

#### Entity Distribution Heatmap
![Entity Heatmap](figures/entity_heatmap.png)

**Insights:**
- **Cluster 2**: Highest gene count (large-scale GWAS)
- **Cluster 5**: Balanced variant/gene/disease (well-structured abstracts)
- **Cluster 3 & 4**: Low entity counts (narrative texts)

---

## Limitations & Error Analysis

### 1. Entity Extraction Challenges

#### Variant Detection (False Negative Rate: 50%)

**Issue**: 35/70 texts lack extracted variants

**Root Causes:**
1. **Conceptual Papers**: Some abstracts discuss genetics without specific rsIDs
   - *Example*: "Pathway analysis reveals novel AD mechanisms..."
2. **Alternative Notations**: Non-rsID variants not captured
   - *Missed*: `chr19:45411941`, `APOE ε4 allele`

**Evidence from Error Analysis:**
```json
{
  "variant_fn_count": 35
}
```

**Next Steps:**
1. Add regex for chromosome positions: `chr\d+:\d+`
2. Map allele names to rsIDs using dbSNP database

---

#### Gene Symbol False Positives (Rate: ~60%)

**Issue**: 287 short tokens + 226 numeric tokens incorrectly labeled as genes

**Examples of False Positives:**
| Category | Examples | Count |
|----------|----------|-------|
| **Abbreviations** | AD (Alzheimer's Disease), PD (Parkinson's), OR (Odds Ratio) | 287 |
| **Numbers** | 10, 437, 6559 (likely sample sizes or p-values) | 226 |
| **Acronyms** | USA, COVID (caught by blacklist), FDR (False Discovery Rate) | ~50 |

**Evidence from Code:**
```python
# From error analysis
'gene_fp_short_count': 287,   # Tokens ≤3 characters
'gene_fp_numeric_count': 226  # Fully numeric tokens
```

**Next Steps:**
1. **Gene Dictionary Validation**: Cross-reference with HGNC gene symbols database
   ```python
   import requests
   hgnc_genes = requests.get("https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json/hgnc_complete_set.json").json()
   valid_symbols = {g['symbol'] for g in hgnc_genes['response']['docs']}
   ```

2. **Use Biomedical NER**: scispaCy's `en_ner_bionlp13cg_md` has 67% precision on gene entities

---

#### Disease/Phenotype Recall (False Negative Rate: 70%)

**Issue**: 49/70 texts fail to extract any disease entities

**Missed Examples:**
| Missed Term | Why Missed | Document ID |
|-------------|-----------|-------------|
| "late-onset Alzheimer's disease" | Too specific (spaCy expects "Alzheimer's disease") | 4 |
| "cognitive decline" | Phenotype, not standard disease label | 11 |
| "hippocampal sclerosis of aging" | Multi-word biomedical term | 19 |

**Evidence:**
```json
{
  "disease_fn_count": 49  
}
```

**Next Steps:**
1. **Switch to scispaCy**: `en_ner_bc5cdr_md` trained on disease mentions
   - Expected improvement: +40% recall
   ```bash
   pip install scispacy
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
   ```

---

### 2. Relation Extraction Over-Generation

#### Triplet Explosion (418 relations from 70 texts)

**Issue**: Cartesian product generates spurious associations

**Example Problem Case:**
```
Document 5 entities:
- Variants: [rs429358, rs7412, rs1081105, rs769449, rs11136000]  # 5
- Genes: [APOE, BIN1, CLU, PICALM, CR1, CD33, MS4A6A, CD2AP, EPHA1, ABCA7]  # 10  
- Diseases: [Alzheimer's disease, late-onset AD, cognitive decline]  # 3

Generated triplets: 5 × 10 × 3 = 150 relations
Actual valid relations in text: ~3
```

**False Positive Example:**
```json
{
    "text_id": 61,
    "variant": "rs6701713",
    "gene": "LOAD",
    "phenotype": "the Alzheimer Disease Genetics Consortium",
    "relation": "associated with",
    "evidence_span": "IMPORTANCE: Because APOE locus variants contribute to risk of late-onset Alzheimer disease (LOAD)"
}
```
*The extracted record for text_id 61 is a clear false positive. The gene field incorrectly identifies LOAD, which is a disease rather than a gene, and the phenotype field incorrectly captures the Alzheimer Disease Genetics Consortium, which is an organization name rather than a clinical phenotype.*

**Current Mitigation:**
- `get_evidence_span()` finds supporting sentences
- But still generates relations when entities are in *different* sentences

**Next Steps:**

1. **Sentence-Level Co-Occurrence Filter**:
   ```python
   def validate_triplet(text, variant, gene, disease):
       doc = nlp(text)
       for sent in doc.sents:
           if variant in sent.text and gene in sent.text and disease in sent.text:
               return True, sent.text
       return False, None
   
   # Expected reduction: 418 → ~120 triplets (-71%)
   ```

---

### 3. Topic Modeling with Small Sample Size

#### Cluster Interpretability Issues

**Statistics:**
- 70 texts ÷ 9 clusters = **7.8 texts/cluster** (avg)
- **Smallest clusters**: 1 text each (Clusters 6, 7) — 2.9% of corpus
- **Largest cluster**: 25 texts (Cluster 2) — 35.7% of corpus
- **Median cluster size**: 4 texts

**Critical Issue: Single-Document Clusters**

Clusters 6 and 7 each contain only **1 document**:
```python
# Evidence from actual data
Cluster 6: 1 text  # "Brain imaging (white matter)"
Cluster 7: 1 text  # "Structural brain measures"
```

**Why This Happens:**
1. **K-Means Forces Assignment**: Even outliers must join a cluster
2. **Unique Vocabulary**: These papers likely use distinct terminology
3. **Potential Causes**:
   - Different imaging modality from other brain studies
   - Methods paper rather than association study
   - Non-AD disease focus (e.g., stroke, MS)

**Impact**: for these 2 clusters
- Keywords are essentially just that document's vocabulary
- No generalization possible
- Should be manually reviewed for:
  - Possible merge with Clusters 1 or 8 (if biomarker-related)
  - Exclusion from corpus (if out of scope)

**Imbalanced Distribution:**
```
Large clusters (>10 texts): 2, 5 → 43 texts (61%)
Small clusters (<5 texts): 0, 3, 4, 6, 7 → 12 texts (17%)
```

This suggests the dataset has **2-3 dominant themes** (AD genetics) with several niche subtopics.

**Next Steps:**

1. **Optimal K Re-evaluation**: 
```python
   # Try k=5 (collapse small clusters)
   silhouette_scores = []
   for k in range(2, 11):
       kmeans = KMeans(n_clusters=k, random_state=42)
       labels = kmeans.fit_predict(embeddings)
       score = silhouette_score(embeddings, labels)
       silhouette_scores.append(score)
   
   # k=5 might be more appropriate for 70 texts
```

2. **Manual Review Required**: 
   - Examine the single document in Cluster 6 and Cluster 7
   - Determine if they're genuinely off-topic or just specialized

3. **Collect More Data**: 
   - Ideal corpus size for 9 stable clusters: **180+ texts** (20 per cluster)
   - For 70 texts, **k=3-5** is more statistically sound

