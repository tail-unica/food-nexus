from statistic import ontology_statistics

ontology_list = ["../csv_file/ontology_hummus.nt"] #, "../csv_file/ontology_hummus_infered.nt"]
output_file = "../csv_file/ontology_statistics_hummus.csv"
ontology_statistics(ontology_list, output_file, type="nt")