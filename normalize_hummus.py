from main import pipeline #type: ignore

input_file = "csv_file/pp_recipes.csv"
output_file = "csv_file/pp_recipes_normalized_by_pipeline.csv"
column_to_normalize = "title"
column_normalized = 'title_normalized'
pipeline(
    input_file=input_file, output_file=output_file, show_something=True, show_all=False, column_name=column_to_normalize, new_column_name=column_normalized, translation_nedded=False,  delimiter=";"
)
