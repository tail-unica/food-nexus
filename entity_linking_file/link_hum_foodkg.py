from entity_linking import merge_embedding

header = "hum_foodkg"
file1 = "../csv_file/hum_recipe_for_linking_embedding.csv"
file2 = "../csv_file/foodkg_recipe_for_linking_embedding.csv"
file_output = "../csv_file/file_hummus_foodkg.csv"
chunk_size = 25000

merge_embedding(header, file1, file2, file_output, chunk_size)