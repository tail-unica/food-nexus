from main import pipeline #type: ignore

file_input = "csv_file/ingredients_food_kg.csv"
file_output = "csv_file/ingredients_food_kg_normalizzed_by_pipeline.csv"
column_to_normalize = "ingredient"
column_normalized = 'ingredient_normalized'
pipeline(
    input_file=file_input, output_file=file_output, show_something=True, show_all=False, column_name=column_to_normalize, new_column_name=column_normalized
)