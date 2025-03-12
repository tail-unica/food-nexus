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

file_off_foodkg = "csv_file/file_off_foodkg.csv"
header = ["Score", "off_recipe", "foodkg_recipe"]
set_off_foodkg = set()
threshold_value = 0.85
model="BAAI/bge-en-icl"
batch_size = 100

### Merge foodkg and off ###
# Columns of the off file to be used for the merging
off_file_path = "csv_file/off_normalized_final_c.csv"
off_column = [
    "product_name",
    "product_name_normalized",
]
list_off_recipe = read_specified_columns(
    off_file_path, off_column, delimiter="\t"
)


list_off_recipe = [row for row in list_off_recipe if row[1] is not None and row[1] != ""]

print("numero ricette off: ", len(list_off_recipe))

# Columns of the foodkg file to be used for the merging
food_kg_path = (
    "csv_file/ingredients_food_kg_normalizzed_by_pipeline.csv"
)
foodkg_column = ["ingredient", "ingredient_normalized"]
list_foodkg = read_specified_columns(
    food_kg_path, foodkg_column, delimiter=","
)

list_foodkg = [row for row in list_foodkg if row[1] is not None and row[1] != ""]

print("numero ricette fkg: ", len(list_foodkg))


chunk_size = 100
total_chunk_off = math.ceil(len(list_off_recipe) / chunk_size)
total_chunk_foodkg = math.ceil(len(list_foodkg) / chunk_size)
is_first = True

print(f"starting merging off to foodkg\n\n")

length_pair = total_chunk_off * total_chunk_foodkg
count_pair = 1
start_time = time.time()

for chunk_count in range(total_chunk_off):

    if chunk_count*(chunk_size+1) >= len(list_off_recipe):
        list_off_temp = list_off_recipe[chunk_count*chunk_size:]
    else:
        list_off_temp = list_off_recipe[chunk_count*chunk_size:(chunk_count+1)*chunk_size]

    for chunk_count2 in range(total_chunk_foodkg):

        if chunk_count2*(chunk_size+1) >= len(list_foodkg):
            list_foodkg_temp = list_foodkg[chunk_count2*chunk_size:]
        else:
            list_foodkg_temp = list_foodkg[chunk_count2*chunk_size:(chunk_count2+1)*chunk_size]

        if count_pair % 1 == 0:
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


        most_similar_pairs = find_k_most_similar_pairs_with_indicators(
            list_off_temp,
            list_foodkg_temp,
            k=-1,
            model=model,
            use_indicator=False,
            batch_size=batch_size
        )

        if is_first:
            mode = "w"
        else:
            mode = "a"

        with open(file_off_foodkg, mode=mode, newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if is_first:
                writer.writerow(header)
                is_first = False

            for score, original_name1, original_name2, _, _ in most_similar_pairs:
                
                if float(score) > threshold_value:
                    
                    element = [score, original_name1, original_name2]

                    if tuple(element) not in set_off_foodkg:
                        set_off_foodkg.add(tuple(element))
                        writer.writerow(element)
