from statistic import ontology_statistics

ontology_list = ["../csv_file/ontology_hummus_not_infered.nt", "../csv_file/ontology_hummus.nt"]
output_file = "../csv_file/ontology_statistics_hummus.csv"
ontology_statistics(ontology_list, output_file, type="nt")