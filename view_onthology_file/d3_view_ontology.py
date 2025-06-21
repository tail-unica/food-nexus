#import argparse
#import json
#import math
#import os
#import random
#import time
#from collections import defaultdict
#
#from rdflib import Graph
#from tqdm import tqdm
#
#
## Load RDF graph
#def load_rdf_graph(file_path):
#    g = Graph()
#    g.parse(file_path, format="turtle")
#    return g
#
#
## Convert RDF graph to nodes and edges
#def convert_rdf_to_graph(
#    graph, include_attributes=True, remove_labels=False, sample_size=1.0
#):
#    nodes = {}
#    edges = []
#    degree = defaultdict(int)  # Track the degree of each node
#
#    # Determine valid nodes (those that serve as subjects in the RDF graph)
#    valid_nodes = set(str(triple[0]).split("/")[-1] for triple in graph)
#
#    n_triples = len(graph)
#    n_sample_triples = round(n_triples * sample_size)
#    sample_triples = set(random.sample(range(n_triples), n_sample_triples))
#    print(f"Sampling {n_sample_triples} triples out of {n_triples} triples")
#    for i, (subj, pred, obj) in tqdm(
#        enumerate(graph), desc="Extracting triples from KG", total=n_triples
#    ):
#        if i not in sample_triples:
#            continue
#
#        # Extract only the final part of the URI
#        subj_str = str(subj).split("/")[-1]
#        pred_str = str(pred).split("/")[-1]
#        obj_str = str(obj).split("/")[-1]
#
#        # Only include objects that are also valid subjects if filtering is required
#        if obj_str not in valid_nodes and not include_attributes:
#            continue
#
#        if subj_str not in nodes:
#            nodes[subj_str] = {
#                "id": subj_str,
#                "label": "" if remove_labels else subj_str,
#                "size": 6,
#                "labelSize": 10,
#            }
#        if obj_str not in nodes:
#            nodes[obj_str] = {
#                "id": obj_str,
#                "label": "" if remove_labels else obj_str,
#                "size": 6,
#                "labelSize": 10,
#            }
#
#        edges.append({"source": subj_str, "target": obj_str, "label": pred_str})
#
#        # Update degree count
#        degree[subj_str] += 1
#        degree[obj_str] += 1
#
#    # Update nodes with size and label size based on degree
#    for node_id, node_data in nodes.items():
#        node_degree = degree[node_id]
#        node_data["size"] = max(6, 2 * math.log2(node_degree + 1))  # Node size
#        node_data["labelSize"] = max(
#            10, 3 * math.log2(node_degree + 1)
#        )  # Label size
#
#    return list(nodes.values()), edges
#
#
#def generate_html(nodes, edges, output_file):
#    html_template = """<!DOCTYPE html>
#<html>
#<head>
#    <meta charset="utf-8">
#    <title>D3.js RDF Graph Visualization</title>
#    <script src="https://d3js.org/d3.v7.min.js"></script>
#    <style>
#        body, html {
#            margin: 0;
#            padding: 0;
#            width: 100%;
#            height: 100%;
#            overflow: hidden;
#        }
#        svg {
#            width: 100%;
#            height: 100%;
#        }
#        .link {
#            stroke-opacity: 0.6;
#        }
#        .node text {
#            pointer-events: none;
#            font-family: sans-serif;
#        }
#    </style>
#</head>
#<body>
#    <script>
#        const nodes = {{ nodes | safe }};
#        const links = {{ edges | safe }};
#
#        const edgeColors = {
#            "memberOf": "#00FF00",  // green
#            "publishesRecipe": "#0000FF",  // blue
#            "publishesReview": "#FF0000",  // red
#            "itemReviewed": "#FF00FF",  // purple
#            "NutritionInformation": "#DDDDDD",  // grey
#            "isSimilarTo": "#FFA500",  // orange
#            "hasPart": "#8B0000",  // dark red
#            "isPartOf": "#FF69B4",  // pink
#            "suitableForDiet": "#008000",  // dark green
#            "default": "#000000"  // black
#        };
#
#        const width = window.innerWidth;
#        const height = window.innerHeight;
#
#        const svg = d3.select("body").append("svg")
#            .attr("width", width)
#            .attr("height", height)
#            .call(d3.zoom().on("zoom", (event) => {
#                g.attr("transform", event.transform);
#            }));
#
#        const g = svg.append("g");
#
#        const simulation = d3.forceSimulation(nodes)
#            .force("link", d3.forceLink(links).id(d => d.id).distance(150))
#            .force("charge", d3.forceManyBody().strength(-300))
#            .force("center", d3.forceCenter(width / 2, height / 2));
#
#        const link = g.append("g")
#            .attr("class", "links")
#          .selectAll("line")
#          .data(links)
#          .enter().append("line")
#            .attr("class", "link")
#            .attr("stroke", d => edgeColors[d.label] || edgeColors["default"])
#            .attr("stroke-width", 2);
#
#        const node = g.append("g")
#            .attr("class", "nodes")
#          .selectAll("g")
#          .data(nodes)
#          .enter().append("g")
#          .call(d3.drag()
#              .on("start", dragstarted)
#              .on("drag", dragged)
#              .on("end", dragended));
#
#        node.append("circle")
#            .attr("r", d => d.size)
#            .attr("fill", "steelblue");
#
#        node.append("text")
#            .attr("x", 12)
#            .attr("dy", ".35em")
#            .style("font-size", d => `${d.labelSize}px`)
#            .text(d => d.label);
#
#        link.append("title").text(d => d.label);
#
#        simulation.on("tick", () => {
#            link
#                .attr("x1", d => d.source.x)
#                .attr("y1", d => d.source.y)
#                .attr("x2", d => d.target.x)
#                .attr("y2", d => d.target.y);
#
#            node
#                .attr("transform", d => `translate(${d.x},${d.y})`);
#        });
#
#        function dragstarted(event, d) {
#            if (!event.active) simulation.alphaTarget(0.3).restart();
#            d.fx = d.x;
#            d.fy = d.y;
#        }
#
#        function dragged(event, d) {
#            d.fx = event.x;
#            d.fy = event.y;
#        }
#
#        function dragended(event, d) {
#            if (!event.active) simulation.alphaTarget(0);
#            d.fx = null;
#            d.fy = null;
#        }
#    </script>
#</body>
#</html>"""
#
#    # Write to file
#    with open(output_file, "w") as f:
#        f.write(
#            html_template.replace(
#                "{{ nodes | safe }}", json.dumps(nodes)
#            ).replace("{{ edges | safe }}", json.dumps(edges))
#        )
#
#
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(
#        description="Convert RDF graph to D3.js visualization"
#    )
#    parser.add_argument(
#        "rdf_file", type=str, default="example.ttl", help="Path to the RDF file"
#    )
#    parser.add_argument(
#        "--include_attributes",
#        action="store_true",
#        help="Include node attributes in the visualization",
#    )
#    parser.add_argument(
#        "--remove_labels",
#        action="store_true",
#        help="Remove labels from nodes in the visualization",
#    )
#    parser.add_argument(
#        "--sample_size",
#        type=float,
#        default=1.0,
#        help="Sample size for the RDF graph",
#    )
#    args = parser.parse_args()
#
#    # Ensure an RDF file is available
#    if not os.path.exists(args.rdf_file):
#        with open(args.rdf_file, "w") as f:
#            f.write(
#                """@prefix ex: <http://example.org/> .
#ex:Subject ex:Predicate ex:Object .
#ex:Object ex:Related ex:AnotherObject ."""
#            )
#
#    print("Loading RDF graph...")
#    load_time = time.time()
#    rdf_graph = load_rdf_graph(args.rdf_file)
#    print("Loaded RDF graph with {} triples".format(len(rdf_graph)), end=" ")
#    load_time = time.time() - load_time
#    print(
#        f"Time taken: {int(load_time // 3600):02d}:{int((load_time % 3600) // 60):02d}:{int(load_time % 60):02d}"
#    )
#    print("Converting RDF graph to nodes and edges...")
#    nodes, edges = convert_rdf_to_graph(
#        rdf_graph,
#        include_attributes=args.include_attributes,
#        remove_labels=args.remove_labels,
#        sample_size=args.sample_size,
#    )
#    output_html = args.rdf_file.replace(".ttl", f"_{args.sample_size}.html")
#    generate_html(nodes, edges, output_html)
#
#    print(f"HTML file generated: {output_html}")
#