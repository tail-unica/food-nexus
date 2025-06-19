from entity_linking import merge_embedding

header = "hum_hum"
file1 = "../csv_file/hum_recipe_for_linking_embedding.csv"
file2 = "../csv_file/hum_recipe_for_linking_embedding.csv"
file_output = "../csv_file/file_hummus_hummus.csv"
chunk_size = 25000
header_2 = ['title_normalized_1', 'title_normalized_2', 'cosine_similarity']

merge_embedding(header, file1, file2, file_output, chunk_size, 0.85, header_2)