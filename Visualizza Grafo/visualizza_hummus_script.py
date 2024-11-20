"""
File che contiene script e dati necessari per la visualizzazione
interattiva dell'ontologia
"""


import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
import itertools
from typing import Callable
import networkx as nx
import requests
import xmltodict
from ipysigma import Sigma
import math
import ipywidgets as widgets
import google.generativeai as genai
import networkx as nx
import numpy as np
import tensorflow as tf
import rdflib

# Carica il path del modulo 
module_path = os.path.abspath("talk-like-a-graph")
if module_path not in sys.path:
    sys.path.insert(0, module_path)


from talk_like_a_graph.graph_generators import generate_graphs
from talk_like_a_graph.graph_tasks import CycleCheck, ShortestPath, NodeCount


# Definizione dei colori degli archi che rappresentano el relazioni nel grafo
edge_colors = {
    "memberOf": "#00FF00",  # green
    "publishesRecipe": "#0000FF",  # blu
    "publishesReview": "#FF0000",  # red
    "itemReviewed": "#FF00FF",  # purple
    "NutritionInformation": "#DDDDDD",  # grey
    "isSimilarTo": "#FFA500",  # orange
    "hasPart": "#8B0000",  # dark red
    "isPartOf": "#FF69B4",  # pink
    "suitableForDiet": "#008000",  # dark green
    "default": "#FFFFFF" #black
}


# Carica il grafo da un file
def load_graph_from_ttl(file_path: str, filter_edges: bool = False) -> nx.Graph:
    """
    funzione per caricare un grafo da un file TTL

    :param file_path: percorso del file TTL
    :param filter_edges: se True, filtra gli archi che non sono presenti nella lista edge_colors
    
    :param return: un grafo di NetworkX 
    """

    # Crea un grafo RDF con rdflib
    rdf_graph = rdflib.Graph()
    rdf_graph.parse(file_path, format="turtle")
    
    # Crea un grafo di NetworkX
    nx_graph = nx.Graph()

    # Itera su tutte le triple (soggetto, predicato, oggetto) e le aggiunge al grafo
    for subj, pred, obj in rdf_graph:
        # Estrae solo la parte finale dell'URI
        subj = str(subj).split('/')[-1]
        pred = str(pred).split('/')[-1]
        obj = str(obj).split('/')[-1]

        # Filtra le relazioni se il flag è impostato
        if filter_edges and pred not in edge_colors:
            continue

        # Aggiunge i nodi
        nx_graph.add_node(subj, label=subj)
        nx_graph.add_node(obj, label=obj)

        # Ottiene il colore dell'arco
        edge_color = edge_colors[pred] if pred in edge_colors else edge_colors["default"]

        # Aggiunge  l'arco con il colore personalizzato
        nx_graph.add_edge(subj, obj, label=pred, edge_color=edge_color)

    return nx_graph