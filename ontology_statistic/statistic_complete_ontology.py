from statistic import ontology_statistics

ontology_list = ["../csv_file/ontology_complete.nt"] 
output_file = "../csv_file/ontology_statistics_complete.csv"
ontology_statistics(ontology_list, output_file, type="nt")