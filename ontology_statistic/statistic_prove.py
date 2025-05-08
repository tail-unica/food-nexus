from statistic import ontology_statistics

ontology_list = ["../csv_file/ontology_prove.nt"] 
output_file = "../csv_file/ontology_statistics_prove.csv"
ontology_statistics(ontology_list, output_file, type="nt")