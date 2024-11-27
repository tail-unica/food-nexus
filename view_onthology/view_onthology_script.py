"""
File containing scripts and data required for the interactive visualization 
of the ontology
"""


import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
import itertools
from typing import Callable
import networkx as nx  # type: ignore
import requests
import xmltodict  # type: ignore
from ipysigma import Sigma  # type: ignore
import ipywidgets as widgets  # type: ignore
import numpy as np
import tensorflow as tf
import rdflib  # type: ignore

# Load the module path
module_path = os.path.abspath("talk-like-a-graph")
if module_path not in sys.path:
    sys.path.insert(0, module_path)


from talk_like_a_graph.graph_generators import generate_graphs  # type: ignore
from talk_like_a_graph.graph_tasks import CycleCheck, ShortestPath, NodeCount  # type: ignore


# Definition of edge colors representing relationships in the graph
edge_colors = {
    "memberOf": "#00FF00",  # green
    "publishesRecipe": "#0000FF",  # blue
    "publishesReview": "#FF0000",  # red
    "itemReviewed": "#FF00FF",  # purple
    "NutritionInformation": "#DDDDDD",  # grey
    "isSimilarTo": "#FFA500",  # orange
    "hasPart": "#8B0000",  # dark red
    "isPartOf": "#FF69B4",  # pink
    "suitableForDiet": "#008000",  # dark green
    "default": "#FFFFFF",  # black
}


# Load the graph from a file
def load_graph_from_ttl(file_path: str, filter_edges: bool = False) -> nx.Graph:
    """
    Function to load a graph from a TTL file

    :param file_path: path to the TTL file
    :param filter_edges: if True, filters out edges not present in the edge_colors list

    :param return: a NetworkX graph
    """

    # Create an RDF graph using rdflib
    rdf_graph = rdflib.Graph()
    rdf_graph.parse(file_path, format="turtle")

    # Create a NetworkX graph
    nx_graph = nx.Graph()

    # Iterate over all triples (subject, predicate, object) and add them to the graph
    for subj, pred, obj in rdf_graph:
        # Extract only the final part of the URI
        subj = str(subj).split("/")[-1]
        pred = str(pred).split("/")[-1]
        obj = str(obj).split("/")[-1]

        # Filter relationships if the flag is set
        if filter_edges and pred not in edge_colors:
            continue

        # Add nodes
        nx_graph.add_node(subj, label=subj)
        nx_graph.add_node(obj, label=obj)

        # Get the edge color
        edge_color = (
            edge_colors[pred] if pred in edge_colors else edge_colors["default"]
        )

        # Add the edge with the custom color
        nx_graph.add_edge(subj, obj, label=pred, edge_color=edge_color)

    return nx_graph
