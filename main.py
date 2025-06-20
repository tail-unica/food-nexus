"""
Script for merging the two ontologies
"""


import os
import sys
import time
import csv
from rdflib import RDF, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL
from rdflib import OWL
from csv import writer
from normalization_pipeline_file.pipeline import pipeline_core, pipeline  # type: ignore
from attribute_extraction_file.attribute_extraction import add_user_attributes  # type: ignore



def add_to_sys_path(folder_name) -> None:
    """
    Function to add a folder to the system path
    :param folder_name: name of the folder to add
    :return: None
    """

    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), folder_name)
    )
    sys.path.append(utils_path)


add_to_sys_path("entity_linking_file")
add_to_sys_path("create_rdf_file")
add_to_sys_path("normalization_pipeline_file")
add_to_sys_path("attribute_extraction_file")


from entity_linking import (  # type: ignore
    read_specified_columns,
    normalize_columns,
    find_k_most_similar_pairs_with_indicators,
)

from create_rdf_file.create_rdf import sanitize_for_uri  # type: ignore
from pipeline import pipeline_core, pipeline  # type: ignore
from attribute_extraction import add_user_attributes  # type: ignore



def create_completed_ontology_streaming() -> None:
    """
    Function to create the hummus-off merged ontology
    :return: None
    """

    
    file_path1 = "./csv_file/ontology_hummus_infered.nt"
    file_path2 = "./csv_file/ontology_off.nt"
    file_path3 = "./csv_file/ontology_merge_off_hum.nt"
    file_path4 = "./csv_file/ontology_merge_off_fkg.nt"
    file_path5 = "./csv_file/ontology_merge_hum_hum.nt"
    output_file = "./csv_file/ontology_complete.nt"
    file_paths = [file_path1, file_path2, file_path3, file_path4, file_path5]

    link_ontology = "https://github.com/tail-unica/kgeats/unica_complete_food_ontology"
    ontology_iri = URIRef(link_ontology)
    version_iri = URIRef(f"{link_ontology}/1.0")
    version_info = Literal("Version 1.0 - Initial release", lang="en")
    prior_version_iri = URIRef(f"{link_ontology}/0.0")

    print(f"Start of merge process. Output file: {output_file}")

    header_triples = [
        f"{ontology_iri.n3()} <{RDF.type}> <{OWL.Ontology}> .",
        f"{ontology_iri.n3()} <{OWL.versionIRI}> {version_iri.n3()} .",
        f"{ontology_iri.n3()} <{OWL.versionInfo}> {version_info.n3()} .",
        f"{ontology_iri.n3()} <{OWL.priorVersion}> {prior_version_iri.n3()} ."
    ]

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            
            print("Writing the ontology header...")
            for triple_line in header_triples:
                outfile.write(triple_line + '\n')
            print("Header written.")

            for i, file_path in enumerate(file_paths):
                print(f"Processing file {i+1}/{len(file_paths)}: {file_path}...")
                if not os.path.exists(file_path):
                    print(f"File not found")
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        line_count = 0
                        for line in infile:
                            outfile.write(line.strip() + '\n')
                            line_count += 1
                        
                            if line_count % 1000000 == 0:
                               print(f"  ... read {line_count // 1000000}M rows from {os.path.basename(file_path)}")

                    print(f"Processing completed of {file_path} ({line_count} rows read).")
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

            print(f"Merge completed. Output has been saved to: {output_file}")

    except IOError as e:
        print(f"Error opening or writing output file {output_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
