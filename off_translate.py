from main import pipeline #type: ignore

input_file = "csv_file/off.csv"
output_file = "csv_file/off_english.csv"
column_to_normalize = "product_name"
column_normalized = 'product_name_english'
pipeline(
    input_file=input_file, output_file=output_file, show_something=True, show_all=False, column_name=column_to_normalize, delimiter="\t", new_column_name=column_normalized, translation_nedded = True, only_translation = True
)
