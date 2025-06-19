from entity_linking import merge_embedding

header = "off_foodkg"
file1 = "../csv_file/off_recipe_for_linking_embedding.csv"
file2 = "../csv_file/foodkg_recipe_for_linking_embedding.csv"
file_output = "../csv_file/file_off_foodkg.csv"
chunk_size = 25000
header_2 = ['product_name_normalized', 'ingredient_normalized', 'cosine_similarity']

merge_embedding(header, file1, file2, file_output, chunk_size, 0.85, header_2)