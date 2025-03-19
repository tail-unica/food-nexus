from entity_linking import calcolate_embeddings

header = ["name_normalized", "embedding"]
file_input = "../csv_file/hum_recipe_for_linking.csv"
file_output = "../csv_file/hum_recipe_for_linking_embedding.csv"
chunk_size = 1000
batch_size = 1000
model1 = "BAAI/bge-en-icl"

calcolate_embeddings(header, file_input, file_output, chunk_size, batch_size, model1)