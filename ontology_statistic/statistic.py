"""
File for calcolate the statistic on a list of ontology
"""


from collections import Counter
from rdflib import Graph, Literal, RDF
import csv


def analyze_turtle_file(file_path) -> dict:
    """
    Analyze a Turtle file for ontology statistics, including counts of instances.

    :param file_path: path to the Turtle file
    :return: a dictionary containing the statistics
    """
    g = Graph()
    g.parse(file_path, format="turtle")

    # Entities and relations
    entities = set()
    relations = set()
    attributes = set()
    entity_types = Counter()
    relation_types = Counter()

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


def ontology_statistics(turtle_files, output_csv) -> None:
    """
    Analyze Turtle files and write statistics to a CSV, including instance counts.

    :params turtle_files: list of paths to Turtle files
    :params output_csv: path to the output CSV file
    """
    all_entity_types = Counter()
    all_relation_types = Counter()

    results = []
    for file_path in turtle_files:
        stats = analyze_turtle_file(file_path)
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
