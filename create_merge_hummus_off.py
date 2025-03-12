import os
import sys
import time
import csv
import math

def add_to_sys_path(folder_name) -> None:
    """
    Function to add a folder to the system path
    :param folder_name: name of the folder to add
    :return: None
    """

    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), folder_name)
    )
    sys.path.append(utils_path)


add_to_sys_path("entity_linking_file")
add_to_sys_path("normalization_pipeline_file")
add_to_sys_path("attribute_extraction_file")


from entity_linking import (  # type: ignore
    read_specified_columns,
    normalize_columns,
    find_k_most_similar_pairs_with_indicators,
)

print(f"starting the merging process\n\n")

file_off_hummus = "csv_file/file_off_hummus.csv"
header = ["Score", "hummus_recipes", "off_recipe", "hummus_id", "off_id"]
threshold_value = 0.85
model="BAAI/bge-en-icl"
batch_size = 100

# Columns of the hummus file to be used for the merging
hummus_file_path = "csv_file/pp_recipes_normalized_by_pipeline_row.csv"
hummus_column: list[str] = [
    "title",
    "totalFat [g]",
    "totalCarbohydrate [g]",
    "protein [g]",
    "servingSize [g]",
    "recipe_id",
    "title_normalized"
]
list_hummus_recipe = read_specified_columns(
    hummus_file_path, hummus_column, delimiter=";"
)

# Normalize the indicator by dividing them by serving size
list_hummus_recipe = normalize_columns(list_hummus_recipe)

list_hummus_recipe = list_hummus_recipe[0:100]

list_hummus_recipe = [row for row in list_hummus_recipe if row[-1] is not None and row[-1] != ""]

print("numero ricette hummus: ", len(list_hummus_recipe))

# Columns of the off file to be used for the merging
off_file_path = "csv_file/off_normalized_final_c.csv"
off_column = [
    "product_name",
    "fat_100g",
    "carbohydrates_100g",
    "proteins_100g",
    "code",
    "product_name_normalized",
]
list_off_recipe = read_specified_columns(
    off_file_path, off_column, delimiter="\t"
)

list_off_recipe = list_off_recipe[0:100]

list_off_recipe = [row for row in list_off_recipe if row[-1] is not None and row[-1] != ""]

print("numero ricette off: ", len(list_off_recipe))

chunk_size = 100
total_chunk_off = math.ceil(len(list_off_recipe) / chunk_size)
total_chunk_hum = math.ceil(len(list_hummus_recipe) / chunk_size)
is_first = True

print(f"starting merging off to hummus\n\n")

length_pair = total_chunk_off * total_chunk_hum
count_pair = 1
start_time = time.time()

for chunk_count in range(total_chunk_off):

    if chunk_count*(chunk_size+1) >= len(list_off_recipe):
        list_off_temp = list_off_recipe[chunk_count*chunk_size:]
    else:
        list_off_temp = list_off_recipe[chunk_count*chunk_size:(chunk_count+1)*chunk_size]

    for chunk_count2 in range(total_chunk_hum):

        if chunk_count2*(chunk_size+1) >= len(list_hummus_recipe):
            list_hum_temp = list_hummus_recipe[chunk_count2*chunk_size:]
        else:
            list_hum_temp = list_hummus_recipe[chunk_count2*chunk_size:(chunk_count2+1)*chunk_size]
            
        if count_pair % 5 == 0:
            elapsed_time: float = time.time() - start_time
            avg_time_per_row = elapsed_time / count_pair
            remaining_time = avg_time_per_row * (length_pair - count_pair)
            days, remainder = divmod(int(remaining_time), 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            count_pair += 1

            print(
                f"\n\n(off-fkg) Processed {count_pair}/{length_pair} chunks. Estimated time remaining: {days} days, {hours} hours, {minutes} minutes\n\n",
                end="\r"
            )

        ### Merge hummus and off ###
        most_similar_pairs = find_k_most_similar_pairs_with_indicators(
            list_hum_temp,
            list_off_temp,
            k=-1,
            model=model,
            use_indicator=True,
            batch_size = batch_size
        )

        if is_first:
            mode = "w"
        else:
            mode = "a"

        with open(file_off_hummus, mode=mode, newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if is_first:
                writer.writerow(header)
                is_first = False

            for score, original_name1, original_name2, id1, id2 in most_similar_pairs:

                if float(score) > threshold_value:
                    element = [score, original_name1, original_name2, id1, id2]
                    writer.writerow(element)




