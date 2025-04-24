from entity_linking import merge_embedding

header = "fkg_fkg"
file1 = "../csv_file/foodkg_recipe_for_linking_embedding.csv"
file2 = "../csv_file/foodkg_recipe_for_linking_embedding.csv"
file_output = "../csv_file/file_foodkg_foodkg.csv"
chunk_size = 100000
model1 = "BAAI/bge-en-icl"

merge_embedding(header, file1, file2, file_output, chunk_size)