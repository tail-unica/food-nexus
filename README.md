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

1. **File Generation**
   Navigate to the appropriate subdirectories containing Jupyter notebooks for data processing and LLM-based tasks. Run the notebooks to generate intermediate representations.

2. **Normalization and Attribute Extraction**
   Run the notebook `pipeline.ipynb` located in the root directory to normalize the datasets and infer user-specific attributes.

3. **Ontology Merging**
   Execute the final notebook `main.ipynb` to merge the extracted knowledge into a unified ontology.

## 📂 Repository Structure (optional)

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

## 🎯 Research Objectives (optional)

This project investigates:

* The effectiveness of LLMs for ontology population and schema alignment.
* Automated semantic enrichment from food-related databases.
* Personalized food knowledge graphs through user-driven attribute inference.

The outcomes are relevant for personalized nutrition, semantic search, and knowledge graph completion in the food domain.

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
