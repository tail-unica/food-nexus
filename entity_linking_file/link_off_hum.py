from entity_linking import merge_embedding

header = "off_hummus"
file1 = "../csv_file/off_recipe_for_linking_embedding.csv"
file2 = "../csv_file/hum_recipe_for_linking_embedding.csv"
file_output = "../csv_file/file_off_hummus.csv"
chunk_size = 40000

merge_embedding(header, file1, file2, file_output, chunk_size)