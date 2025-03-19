from entity_linking import merge_embedding

header = ["name_normalized", "embedding"]
file1 = "../csv_file/off_recipe_for_linking_embedding.csv"
file2 = "../csv_file/hum_recipe_for_linking_embedding.csv"
file_output = "../csv_file/file_off_hummus.csv"
chunk_size = 100
batch_size = 100
model1 = "BAAI/bge-en-icl"

merge_embedding(header, file1, file2, file_output, chunk_size, batch_size, model1)