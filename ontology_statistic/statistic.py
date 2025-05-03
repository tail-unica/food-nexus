"""
File for calcolate the statistic on a list of ontology
"""

from collections import Counter
from rdflib import Graph, Literal, RDF
import csv
import os
import gc 
import time

def analyze_turtle_file(file_path) -> dict:
    """
    Analyze a Turtle file for ontology statistics, including counts of instances.

    :param file_path: path to the Turtle file
    :return: a dictionary containing the statistics
    """

    # Entities and relations
    entities = set()
    relations = set()
    attributes = set()
    entity_types = Counter()
    relation_types = Counter()

    g = Graph()
    g.parse(file_path, format="turtle")

    # Analyze the RDF triples
    for subject, predicate, obj in g:
        # Collect entities and relations
        entities.add(subject)
        relations.add(predicate)
        if isinstance(obj, Literal):
            attributes.add(obj)
        else:
            entities.add(obj)

        # Count the instances of entity types
        if predicate == RDF.type:
            entity_types[str(obj)] += 1

        # Count the relations
        relation_types[str(predicate)] += 1

    return {
        "num_triples": len(g),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_attributes": len(attributes),
        "num_entity_types": len(entity_types),
        "num_relation_types": len(relation_types),
        "entity_types": entity_types,
        "relation_types": relation_types,
    }


def analyze_nt_file(file_path, batch_size=10000):
    """
    Analyze an NT (N-Triples) file in batches for ontology statistics.
    
    :param file_path: path to the NT file
    :param batch_size: number of triples to process in each batch
    :return: a dictionary containing the aggregated statistics
    """

    # Entities and relations
    entities = set()
    relations = set()
    attributes = set()
    entity_types = Counter()
    relation_types = Counter()
    
    total_triples = 0
    batch_count = 0
    total_lines = sum(1 for _ in open(file_path, encoding="utf-8")) - 1
    total_chunks = total_lines/batch_size
    numchunk = 0
    start_total = time.time()

    # Process the file in batches
    with open(file_path, 'r', encoding='utf-8') as file:
        current_batch = []

        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            current_batch.append(line)
            
            # Process batch when it reaches the specified size
            if len(current_batch) >= batch_size:
                chunk_start = time.time()
                batch_stats = process_batch(current_batch, entities, relations, attributes, 
                                          entity_types, relation_types)
                total_triples += batch_stats["batch_triples"]
                print(f"Processed batch {numchunk+1} of {round(total_chunks)}, file {file_path}")
                current_batch = []
                
                chunk_time = time.time() - chunk_start
                avg_time_per_chunk = (time.time() - start_total) / (numchunk + 1)
                remaining_chunks = total_chunks - (numchunk + 1)
                est_remaining = avg_time_per_chunk * remaining_chunks
                print(f"Chunk time: {chunk_time:.2f}s — Estimated remaining: {est_remaining/60:.1f} min")
                numchunk += 1


        # Process any remaining triples in the last batch
        if current_batch:
            batch_stats = process_batch(current_batch, entities, relations, attributes, 
                                      entity_types, relation_types)
            total_triples += batch_stats["batch_triples"]
            
            batch_count += 1
            print(f"Processed final batch {batch_count}: {batch_stats['batch_triples']} triples, "
                  f"Total: {total_triples} triples, ")


    print(f"Analysis completed in {time.time() - start_total:.2f} seconds.")
    
    return {
        "num_triples": total_triples,
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_attributes": len(attributes),
        "num_entity_types": len(entity_types),
        "num_relation_types": len(relation_types),
        "entity_types": entity_types,
        "relation_types": relation_types,
    }

def process_batch(batch_lines, entities, relations, attributes, entity_types, relation_types):
    """
    Process a batch of NT lines and update the statistics.
    
    :param batch_lines: list of NT lines to process
    :param entities: set of entities to update
    :param relations: set of relations to update
    :param attributes: set of attributes to update
    :param entity_types: Counter for entity types
    :param relation_types: Counter for relation types
    :return: statistics for this batch
    """
    g = Graph()
    
    # Parse the batch of triples
    nt_content = '\n'.join(batch_lines)
    g.parse(data=nt_content, format="nt")
    
    batch_triples = 0
    
    # Analyze the RDF triples in this batch
    for subject, predicate, obj in g:
        batch_triples += 1
        
        # Collect entities and relations
        entities.add(subject)
        relations.add(predicate)
        
        if isinstance(obj, Literal):
            attributes.add(obj)
        else:
            entities.add(obj)
        
        # Count the instances of entity types
        if predicate == RDF.type:
            entity_types[str(obj)] += 1
        
        # Count the relations
        relation_types[str(predicate)] += 1
    
    return {"batch_triples": batch_triples}




def ontology_statistics(turtle_files, output_csv, type="nt") -> None:
    """
    Analyze Turtle files and write statistics to a CSV, including instance counts.

    :params turtle_files: list of paths to Turtle files
    :params output_csv: path to the output CSV file
    """
    all_entity_types = Counter()
    all_relation_types = Counter()

    results = []
    for file_path in turtle_files:

        if type == "ttl":
            stats = analyze_turtle_file(file_path)
        else:
            stats = analyze_nt_file(file_path)

        
        results.append(
            {
                "file": file_path.split("/")[-1],
                "num_triples": stats["num_triples"],
                "num_entities": stats["num_entities"],
                "num_relations": stats["num_relations"],
                "num_attributes": stats["num_attributes"],
                "num_entity_types": len(stats["entity_types"]),
                "num_relation_types": len(stats["relation_types"]),
                "entity_types": stats["entity_types"],
                "relation_types": stats["relation_types"],
            }
        )
        all_entity_types.update(stats["entity_types"])
        all_relation_types.update(stats["relation_types"])

    entity_type_cols = sorted(all_entity_types.keys())
    relation_type_cols = sorted(all_relation_types.keys())

    header = (
        [
            "file",
            "num_triples",
            "num_entities",
            "num_relations",
            "num_attributes",
            "num_entity_types",
            "num_relation_types",
        ]
        + [
            f"entity_type_{entita.split('/')[-1]}"
            for entita in entity_type_cols
        ]
        + [
            f"relation_type_{relazione.split('/')[-1]}"
            for relazione in relation_type_cols
        ]
    )

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()

        for result in results:
            row = {
                "file": result["file"],
                "num_triples": result["num_triples"],
                "num_entities": result["num_entities"],
                "num_relations": result["num_relations"],
                "num_attributes": result["num_attributes"],
                "num_entity_types": result["num_entity_types"],
                "num_relation_types": result["num_relation_types"],
            }
            # Add entity counts
            for entity_type in entity_type_cols:
                row[f"entity_type_{entity_type.split('/')[-1]}"] = result[
                    "entity_types"
                ].get(entity_type, 0)
            # Add relation counts
            for relation_type in relation_type_cols:
                row[f"relation_type_{relation_type.split('/')[-1]}"] = result[
                    "relation_types"
                ].get(relation_type, 0)

            writer.writerow(row)
