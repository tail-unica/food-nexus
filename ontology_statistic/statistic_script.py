from statistic import ontology_statistics

ontology_list = ["../csv_file/ontology_hummus_not_infered.ttl", "../csv_file/ontology_hummus.ttl",  "../csv_file/ontology_off.ttl"]# ,  "../csv_file/unica_food_ontology.ttl"]
output_file = "../csv_file/ontology_statistics.csv"
ontology_statistics(ontology_list, output_file)