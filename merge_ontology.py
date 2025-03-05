from main import create_completed_ontology

#put merge=True if you want to merge the ontologies
#you can also set a threshold value to merge the ontologies
#you can also set a model to use for the merging
create_completed_ontology(merge=True, threshold_value=0.85, model="BAAI/bge-en-icl")