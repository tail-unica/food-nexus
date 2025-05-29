# KGEats

**KGEats** is a joint repository by *Giovanni Zedda* (internship, 0negip) and *FoodNexus*, developed as part of an applied research initiative in knowledge representation, semantic technologies, and food informatics.

## 📘 Overview

**KGEats** aims to:

* Develop one of the most comprehensive food-related ontologies currently available.
* Investigate novel methodologies for **entity linking** and **user attribute extraction** leveraging Large Language Models (LLMs).
* Integrate heterogeneous food knowledge bases (e.g., *OpenFoodFacts*, *HUMMUS*) into a coherent semantic framework.

This project explores the intersection of ontology engineering, natural language processing, and user modeling in the food domain.

## 📦 Installation and Usage

### 1. Requirements

Ensure you are using **Python ≥ 3.8**. Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Dependencies

To run the scripts successfully, the following external resources are required:

* **HUMMUS**: Preprocessed CSV files (from the `data preprocess` folder in the [HUMMUS repository](https://gitlab.com/felix134/connected-recipe-data-set/-/tree/master/data/hummus_data/preprocessed?ref_type=heads))
* **OFF Ontology**: The OpenFoodFacts ontology, available from the [OFF Wiki](https://world.openfoodfacts.org/data) in csv format.

Place these files in a new folder named `csv_file` at the root of this repository.


### 3. Execution Workflow

This workflow outlines the steps to reproduce the KGEats ontology, from initial data setup to the final merged knowledge graph. Ensure you have completed the steps in "1. Requirements" and "2. Data Dependencies" before proceeding.

1.  **Initial Data Setup:**
    *   Download the required datasets (HUMMUS preprocessed CSVs, OpenFoodFacts CSV) from their official sources, as detailed in the "Data Dependencies" section.
    *   Place all downloaded CSV files into the `csv_file/` directory at the root of this repository.

2.  **Environment and LLM Setup:**
    *   Ensure you have created a Python virtual environment (e.g., using `venv` or `conda`) and installed all dependencies by running `pip install -r requirements.txt` within that environment.
    *   Download and install [Ollama](https://ollama.com/) if you intend to run Large Language Models (LLMs) locally. Follow the official Ollama installation guide.
    *   Create the necessary local LLM models (if using Ollama) by executing the `llm_creation.py` script or the associated Jupyter notebook located in the `analisys_and_file_creation_file/` directory. This step configures the specific LLMs used for inference tasks.

3.  **Core Data Processing and File Generation:**
    *   Execute the `file_creation_jupyter.ipynb` notebook, located in the `analisys_and_file_creation_file/` directory, to perform initial data processing and generate intermediate files.

4.  **Dataset Normalization:**
    *   Run the normalization scripts to standardize recipe data from different sources:
        *   `python normalization_pipeline_file/normalize_hummus.py`
        *   `python normalization_pipeline_file/off_normalize.py`

        These scripts will produce normalized recipe datasets.

5.  **LLM-based Information Inference:**
    *   Execute the following scripts to infer additional information (e.g., user attributes, enhanced descriptions) using the configured LLMs:
        *   `python attribute_extraction_file/infere_info_from_description.py`
        *   `python attribute_extraction_file/infere_info_from_review.py`

6.  **Entity Linking Preparation and Execution:**
    *   **Recipe List Generation:** Execute the *last cell* in the `entity_linking_jupyter.ipynb` notebook, found within the `entity_linking_file/` directory. This step creates a consolidated list of recipes from all datasets, essential for the subsequent BERT-based merging process.
    *   **Embedding Creation:** Generate embeddings for recipe names, which are crucial for similarity-based linking. Execute the following scripts:
        *   `python entity_linking_file/create_embedding_foodkg.py`
        *   `python entity_linking_file/create_embedding_hummus.py`
        *   `python entity_linking_file/create_embedding_off.py`
    *   **Linking Execution:** Perform the entity linking between datasets by running:
        *   `python entity_linking_file/link_off_hum.py`
        *   `python entity_linking_file/link_off_foodkg.py`
        *   `python entity_linking_file/link_hum_hum.py` 
        
    *   **Threshold Filtering:** If a stricter association threshold is desired for recipe linking, use the `filter_association_by_threshold.ipynb` notebook (likely in `entity_linking_file/`). This allows you to refine the linking results (e.g., by setting a similarity threshold of 0.975). You will need to adjust some file name in the next script if you dont do this step

7.  **RDF Graph Generation:**
    *   Convert the processed and linked data into RDF triples. Execute the following scripts, founded in the `create_rdf_file/` directory:
        *   `python create_rdf_file/hummus_to_rdf_not_infered.py`
        *   `python create_rdf_file/hummus_to_rdf.py`
        *   `python create_rdf_file/merge_hum_hum_ontology_to_rdf.py`
        *   `python create_rdf_file/merge_off_fkg_ontology_to_rdf.py`
        *   `python create_rdf_file/merge_off_hum_ontology_to_rdf.py`
        *   `python create_rdf_file/off_to_rdf.py`

8.  **Final Ontology Merging:**
    *   Execute the `merge_ontology.py` script located in the root directory to combine all generated RDF graphs into the final, unified KGEats ontology.

## 📂 Repository Structure

```
KGEats/
├── analisys_and_file_creation_file/  # Scripts and notebooks for data analysis and RDF prep
├── attribute_extraction_file/        # Attribute extraction logic from user
├── config_file/                      # Configuration for preprocessing and unit normalization
├── create_rdf_file/                  # Scripts to generate RDF triples from processed data
├── csv_file/                         # All the non-script file
├── entity_linking_file/              # Scripts for entity linking and embedding generation
├── llm_model_file/                   # Prompt templates for LLMs
├── normalization_pipeline_file/      # Pipeline for normalization of attributes and ingredients
├── ollama_server_file/               # File for using ollama for a long period of time
├── ontology_statistic/               # Ontology evaluation and statistics scripts
├── view_onthology_file/              # Visualization tools for ontologies
├── main.py                           # Main script
├── merge_ontology.py                 # Main script for ontology merging
├── pipeline_jupyter.ipynb            # Notebook for calcolate cohen coefficient
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## 🎯 Research Objectives

This project investigates:

* The effectiveness of LLMs for ontology population and schema alignment.
* Automated semantic enrichment from food-related databases.
* Personalized food knowledge graphs through user-driven attribute inference.

The outcomes are relevant for personalized nutrition, semantic search, and knowledge graph completion in the food domain.

## ⬇️ Download the full dataset 
* [Zenodo link](https://zenodo.org/records/15446860?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjMyNDM5ZTEyLTRlODYtNDljOC04MDI2LTIwNzQ3NDc0NmIxMiIsImRhdGEiOnt9LCJyYW5kb20iOiJlOTA3ZDQ3YzRjYjUyN2YwMDQwMTY4YmRmMzliNTVlZSJ9.OHft7HkLO8JTfgI7pQaB8m9SHMkMdJ71ZQPsIh8oKyj-nZRdZr-KsAisUttCM4EiGeFpk23Q1wQZR1xOJaG0qw)

## 📜 License

This project is released under the **MIT License**.
Please refer to the [LICENSE](LICENSE) file for more information.

## 👥 Authors

* **Giovanni Zedda**

  * [LinkedIn](https://it.linkedin.com/in/giovanni-zedda-99a18231b)
  * [GitHub](https://github.com/0negip)

* **Giacomo Medda**

  * [LinkedIn](https://www.linkedin.com/in/giacomo-medda-4b7047200/)
  * [GitHub](https://github.com/jackmedda)

## 🙏 Acknowledgements

We gratefully acknowledge:

* **OpenFoodFacts** and **HUMMUS** for providing food-related datasets.
* The **rdflib** community for RDF parsing and serialization tools.
* The **University of Cagliari** for academic support and infrastructure.

## 📢 Contact

For questions, feedback, or collaborations, please contact:
📧 [zeddagiovanni4021@gmail.com](mailto:zeddagiovanni4021@gmail.com)
