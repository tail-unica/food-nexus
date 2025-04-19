from statistic import ontology_statistics

ontology_list = ["../csv_file/ontology_hummus_not_infered.nt", "../csv_file/ontology_hummus.nt"]#,  "../csv_file/ontology_off.nt"]# ,  "../csv_file/unica_food_ontology.nt"]
output_file = "../csv_file/ontology_statistics.csv"
ontology_statistics(ontology_list, output_file)