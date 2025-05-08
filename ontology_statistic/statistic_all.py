from statistic import ontology_statistics

ontology_list = ["../csv_file/ontology_hummus.nt", "../csv_file/ontology_hummus_infered.nt", "../csv_file/ontology_off.nt", "../csv_file/ontology_complete.nt"]
output_file = "../csv_file/ontology_statistics_all.csv"
ontology_statistics(ontology_list, output_file, type="nt")