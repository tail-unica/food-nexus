from entity_linking import calcolate_embeddings

header = ["name_normalized", "embedding"]
file_input = "../csv_file/off_recipe_for_linking.csv"
file_output = "../csv_file/off_recipe_for_linking_embedding.csv"
chunk_size = 50
batch_size = 50
model1 = "BAAI/bge-en-icl"

calcolate_embeddings(header, file_input, file_output, chunk_size, batch_size, model1)
