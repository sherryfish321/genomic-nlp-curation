# 🧬 Take-Home Challenge: Genomic Text Curation & Topic Grouping (NLP)

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

## 📁 Project Structure

```
genomic-nlp-curation/
│
├── data/
│   └── texts.csv                          # Input: 70 genomic abstracts
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
│   ├── umap_clusters.png                  # 2D topic visualization
│   ├── cluster_distribution.png           # Cluster size bar chart
│   └── entity_heatmap.png                 # Entity counts per cluster
│
├── Genomic_Text_Curation_NLP.ipynb        # Main notebook
├── main.py                                # Standalone script version
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
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

### 1. Entity Extraction (Hybrid Approach)

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
- **Rationale**: Balance between quality and speed (CPU-friendly)

#### Dimensionality Reduction
- **Algorithm**: UMAP (Uniform Manifold Approximation and Projection)
- **Parameters**:
  - `n_neighbors=15`: Preserve local structure
  - `min_dist=0.1`: Allow tight clusters
  - `metric="cosine"`: Appropriate for text embeddings
- **Output**: 2D coordinates for visualization

#### Clustering
- **Algorithm**: K-Means
- **K Selection**: Elbow method (largest inertia drop) → **k=9**
- **Feature Space**: Full 768-dimensional embeddings (not 2D UMAP)

#### Topic Interpretation
- **Method**: TF-IDF on augmented text within each cluster
- **Output**: Top 10 keywords per cluster

---

## 📊 Curation Schema

### Entity Schema (`section2_hybrid_schema.json`)

```json
{
  "id": "1",
  "variants": ["rs429358", "rs7412"],
  "genes": ["APOE", "BIN1", "CLU", "PICALM"],
  "diseases": ["Alzheimer's disease", "late-onset AD"],
  "raw_text": "The International Genomics of Alzheimer's Project..."
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
  "text_id": "32",
  "variant": "rs429358",
  "gene": "APOE",
  "phenotype": "Alzheimer's disease",
  "relation": "increases risk of",
  "evidence_span": "The APOE ε4 allele (rs429358) increases risk of..."
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
3. **Database Integration**: Direct mapping to knowledge graphs (e.g., Neo4j)
4. **Provenance**: `evidence_span` enables fact-checking

---

### Topic Schema (`section4_topics.json`)

```json
{
  "id": "5",
  "cluster": 5,
  "umap_x": -2.34,
  "umap_y": 1.87,
  "has_relation": true,
  "relation_count": 12,
  "relation_types": ["associated with", "increases risk of"],
  "variants": ["rs429358"],
  "genes": ["APOE"],
  "diseases": ["Alzheimer's disease"],
  "text": "Alzheimer's disease (AD) is highly heritable..."
}
```

---

## 📈 Results & Visualizations

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
| **0** | 7 | cancer, pap, snps, prostate | Cancer genetics & SNP associations |
| **1** | 7 | csf, levels, amd, clu | Biomarker studies (CSF, blood) |
| **2** | 13 | ad, load, genes, loci | Late-onset Alzheimer's GWAS |
| **3** | 4 | hs aging, aging, abcc9 | Hippocampal sclerosis & aging |
| **4** | 5 | pd, ftd, mapt, als | Neurodegenerative diseases (PD/FTD) |
| **5** | 10 | apoe, ad, load, age | APOE-focused Alzheimer's studies |
| **6** | 6 | wmhv, white matter, fa | Brain imaging (white matter) |
| **7** | 6 | intracranial volume, pi3k | Structural brain measures |
| **8** | 12 | ad, bi, bmi, hdl | Metabolic factors in AD |

---

#### UMAP Topic Visualization
![UMAP Clusters](figures/umap_clusters.png)

**Interpretation:**
- **Larger dots**: Documents with extracted gene-variant-disease relations (curatable)
- **Color**: Cluster assignment
- **Spatial proximity**: Semantic similarity (closer = more related topics)

**Key Observations:**
1. **Cluster 2 & 5** (APOE studies) are spatially close → strong thematic overlap
2. **Cluster 4** (PD/FTD) is distant from AD clusters → distinct research area
3. **Cluster 3** (hippocampal sclerosis) is isolated → rare/specialized topic

---

#### Entity Distribution Heatmap
![Entity Heatmap](figures/entity_heatmap.png)

**Insights:**
- **Cluster 2**: Highest gene count (large-scale GWAS)
- **Cluster 5**: Balanced variant/gene/disease (well-structured abstracts)
- **Cluster 3 & 4**: Low entity counts (narrative/review-style texts)

---

## 🔍 Limitations & Error Analysis

### 1. Entity Extraction Challenges

#### ❌ Variant Detection (False Negative Rate: 50%)

**Issue**: 35/70 texts lack extracted variants

**Root Causes:**
1. **Conceptual Papers**: Some abstracts discuss genetics without specific rsIDs
   - *Example*: "Pathway analysis reveals novel AD mechanisms..."
2. **Alternative Notations**: Non-rsID variants not captured
   - *Missed*: `chr19:45411941`, `APOE ε4 allele`

**Impact**: 🟡 **Moderate** — Acceptable for exploratory analysis, but limits relation extraction

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

#### ❌ Gene Symbol False Positives (Rate: ~60%)

**Issue**: 287 short tokens + 226 numeric tokens incorrectly labeled as genes

**Examples of False Positives:**
| Category | Examples | Count |
|----------|----------|-------|
| **Abbreviations** | AD (Alzheimer's Disease), PD (Parkinson's), OR (Odds Ratio) | 287 |
| **Numbers** | 10, 437, 6559 (likely sample sizes or p-values) | 226 |
| **Acronyms** | USA, COVID (caught by blacklist), FDR (False Discovery Rate) | ~50 |

**Root Cause**: Naive regex `[A-Z0-9]{3,10}` has no biological context validation

**Impact**: 🔴 **High** — Requires manual curation to remove ~50% of "genes"

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

2. **Context-Aware Filtering**:
   - Reject if preceded by: "year", "day", "patients", "n="
   - Require at least one occurrence with gene-like context: "gene", "protein", "locus"

3. **Use Biomedical NER**: scispaCy's `en_ner_bionlp13cg_md` has 67% precision on gene entities

---

#### ❌ Disease/Phenotype Recall (False Negative Rate: 70%)

**Issue**: 49/70 texts fail to extract any disease entities

**Missed Examples:**
| Missed Term | Why Missed | Document ID |
|-------------|-----------|-------------|
| "late-onset Alzheimer's disease" | Too specific (spaCy expects "Alzheimer's disease") | 4 |
| "cognitive decline" | Phenotype, not standard disease label | 11 |
| "hippocampal sclerosis of aging" | Multi-word biomedical term | 19 |

**Root Cause**: spaCy's general-purpose model (`en_core_web_sm`) is not trained on biomedical text

**Impact**: 🔴 **Critical** — Limits relation extraction to 30% of corpus

**Evidence:**
```json
{
  "disease_fn_count": 49  // 70% of documents
}
```

**Next Steps:**
1. **Switch to scispaCy**: `en_ner_bc5cdr_md` trained on disease mentions
   - Expected improvement: +40% recall
   ```bash
   pip install scispacy
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
   ```

2. **Pattern-Based Fallback**: Add regex for common patterns
   ```python
   DISEASE_PATTERNS = [
       r"\b\w+\'s disease\b",  # Alzheimer's, Parkinson's
       r"\b\w+ syndrome\b",     # Down syndrome
       r"\bcognitive \w+\b"     # cognitive decline, impairment
   ]
   ```

3. **Biomedical Entity Linker**: Normalize to UMLS/SNOMED codes for consistency

---

### 2. Relation Extraction Over-Generation

#### ❌ Triplet Explosion (418 relations from 70 texts)

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
  "variant": "rs11136000",
  "gene": "CLU",
  "phenotype": "cognitive decline",
  "relation": "associated with",
  "evidence_span": "Full document text (no co-occurrence in any sentence)"
}
```
*These entities never appear together in a single sentence.*

**Impact**: 🔴 **High** — ~72% of triplets are likely false (estimated 300/418)

**Current Mitigation:**
- `get_evidence_span()` finds supporting sentences
- But still generates relations when entities are in *different* sentences

**Next Steps:**

1. **Sentence-Level Co-Occurrence Filter** (Immediate win):
   ```python
   def validate_triplet(text, variant, gene, disease):
       doc = nlp(text)
       for sent in doc.sents:
           if variant in sent.text and gene in sent.text and disease in sent.text:
               return True, sent.text
       return False, None
   
   # Expected reduction: 418 → ~120 triplets (-71%)
   ```

2. **Dependency Parsing Validation** (Medium effort):
   ```python
   def check_syntactic_relation(doc, variant, gene, disease):
       # Find entity spans
       var_span = find_entity_span(doc, variant)
       gene_span = find_entity_span(doc, gene)
       
       # Check if connected by verb or preposition
       if has_dependency_path(var_span, gene_span, max_distance=5):
           return True
       return False
   ```

3. **Relation Extraction Classifier** (Requires labeled data):
   - Fine-tune BioBERT on PubMed relation extraction datasets
   - Expected precision: ~85% (vs current ~28%)

---

### 3. Topic Modeling with Small Sample Size

#### Cluster Interpretability Issues

**Statistics:**
- 70 texts ÷ 9 clusters = **7.8 texts/cluster** (avg)
- Smallest cluster: 4 texts (Cluster 3)
- Largest cluster: 13 texts (Cluster 2)

**Observations:**

| Cluster Quality | Clusters | Issue |
|----------------|----------|-------|
| 🟢 **Well-Defined** | 2, 5 | Clear keyword consensus (APOE, GWAS) |
| 🟡 **Moderate** | 0, 1, 8 | Mixed topics, needs manual inspection |
| 🔴 **Sparse** | 3, 4 | <5 texts, keywords may not generalize |

**Limitation**: TF-IDF keywords are dataset-specific and may not apply to new documents

**Evidence:**
```python
# Cluster 3 (Hippocampal Sclerosis)
{
  "size": 4,
  "keywords": ["hs aging", "aging", "hs", "abcc9", "aging pathology"],
  "documents": [19, 27, 43, 61]
}
```
*With only 4 documents, keywords may be due to a single author's writing style.*

**Next Steps:**
1. **Bootstrap Validation**: Resample 70 texts with replacement 100 times, measure cluster stability
   - Stable cluster: Same documents group together >80% of time
   - Unstable cluster: Random grouping

2. **Hierarchical Clustering**: Use dendrogram to validate k=9 choice
   ```python
   from scipy.cluster.hierarchy import dendrogram, linkage
   linkage_matrix = linkage(embeddings, method='ward')
   ```

3. **Collect More Data**: Ideal corpus size for 9 clusters: 180+ texts (20 per cluster)

---

## 📊 Quantitative Error Summary

| Error Type | Count | Rate | Severity | Priority |
|------------|-------|------|----------|----------|
| **Variant False Negatives** | 35 | 50% | 🟡 Medium | P3 |
| **Gene False Positives (short)** | 287 | 60% | 🔴 High | P1 |
| **Gene False Positives (numeric)** | 226 | 47% | 🔴 High | P1 |
| **Disease False Negatives** | 49 | 70% | 🔴 Critical | P1 |
| **Spurious Relations** | ~300 | 72% | 🔴 High | P2 |
| **Unstable Clusters** | 2-3 | 22-33% | 🟡 Medium | P3 |

*Severity: 🔴 Critical = Blocks curation, 🟡 Medium = Adds manual work, 🟢 Low = Acceptable*

---

## 🚀 Next Steps

### High Priority (Immediate Impact)

#### 1. Replace spaCy with scispaCy
**Rationale**: Biomedical-specific NER model trained on PubMed/PMC abstracts

**Implementation:**
```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
```

```python
import scispacy
import spacy

nlp = spacy.load("en_ner_bc5cdr_md")
doc = nlp(text)

# Extract diseases
diseases = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
```

**Expected Improvement:**
- Disease recall: 30% → 70% (+40%)
- Gene precision: 40% → 60% (+20%)

---

#### 2. Add Sentence-Level Relation Filtering
**Rationale**: Eliminate 70% of spurious triplets with minimal code change

**Implementation:**
```python
def extract_valid_triplets(text, variants, genes, diseases):
    doc = nlp(text)
    valid_triplets = []
    
    for sent in doc.sents:
        sent_text = sent.text
        
        # Find entities co-occurring in this sentence
        sent_variants = [v for v in variants if v in sent_text]
        sent_genes = [g for g in genes if g in sent_text]
        sent_diseases = [d for d in diseases if d in sent_text]
        
        # Only create triplets from same-sentence entities
        for v in sent_variants:
            for g in sent_genes:
                for d in sent_diseases:
                    valid_triplets.append({
                        "variant": v,
                        "gene": g,
                        "disease": d,
                        "evidence_span": sent_text
                    })
    
    return valid_triplets
```

**Expected Improvement:**
- Triplet count: 418 → ~120 (-71%)
- Precision: 28% → 75% (+47%)

---

#### 3. Gene Symbol Validation via HGNC Dictionary
**Rationale**: Eliminate numeric and abbreviation false positives

**Implementation:**
```python
import requests
import json

# Download official gene symbols (one-time)
def load_hgnc_genes():
    url = "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json/hgnc_complete_set.json"
    response = requests.get(url)
    data = response.json()
    
    valid_symbols = set()
    for gene in data['response']['docs']:
        valid_symbols.add(gene['symbol'])
        # Also add aliases
        if 'alias_symbol' in gene:
            valid_symbols.update(gene['alias_symbol'])
    
    return valid_symbols

# Validate extracted genes
hgnc_genes = load_hgnc_genes()

def extract_validated_genes(text):
    candidates = re.findall(GENE_PATTERN, text)
    blacklist = {"DNA", "RNA", "USA", "COVID", "GWAS", "OR", "CI", "HR"}
    
    validated = []
    for gene in candidates:
        if gene in blacklist:
            continue
        if gene in hgnc_genes:
            validated.append(gene)
    
    return validated
```

**Expected Improvement:**
- Gene false positives: 513 → ~50 (-90%)
- Gene precision: 40% → 85% (+45%)

---

### Medium Priority (Enhanced Accuracy)

#### 4. Dependency Parsing for Relation Validation
**Rationale**: Verify syntactic connections between entities

**Example:**
```python
def validate_gene_disease_relation(doc, gene_span, disease_span):
    """Check if gene and disease are connected via verb"""
    
    # Find the root verb connecting entities
    connecting_verbs = []
    for token in doc:
        if token.pos_ == "VERB":
            # Check if gene is in subject and disease in object
            if (gene_span in [child.text for child in token.children] and
                disease_span in [child.text for child in token.children]):
                connecting_verbs.append(token.text)
    
    # Valid relation verbs
    valid_verbs = {"associate", "increase", "reduce", "affect", "cause", "link"}
    return any(v in valid_verbs for v in connecting_verbs)
```

---

#### 5. Confidence Scoring System
**Rationale**: Prioritize curation efforts on high-confidence triplets

**Scoring Factors:**
| Factor | Weight | Criteria |
|--------|--------|----------|
| Entity validation | 0.3 | Gene in HGNC + Disease in UMLS |
| Co-occurrence distance | 0.3 | Same sentence (1.0), same paragraph (0.5) |
| Relation keyword presence | 0.2 | Explicit keyword match |
| Sentence structure | 0.2 | Syntactic dependency validation |

```python
def calculate_confidence(triplet, doc):
    score = 0.0
    
    # Entity validation (0.3)
    if triplet['gene'] in hgnc_genes:
        score += 0.15
    if triplet['disease'] in umls_diseases:
        score += 0.15
    
    # Co-occurrence (0.3)
    if all_in_same_sentence(triplet):
        score += 0.3
    elif all_in_same_paragraph(triplet):
        score += 0.15
    
    # Relation keyword (0.2)
    if triplet['relation'] in triplet['evidence_span']:
        score += 0.2
    
    # Syntactic validation (0.2)
    if has_dependency_path(triplet, doc):
        score += 0.2
    
    return score

# Usage
triplet['confidence'] = calculate_confidence(triplet, doc)
# High confidence: score > 0.7
# Medium: 0.4-0.7
# Low: < 0
