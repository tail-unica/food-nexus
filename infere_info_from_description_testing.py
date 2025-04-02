from main import add_user_attributes #type: ignore

column_name = "member_description"
new_column_names = ["weight", "height", "gender", "age", "user_constraints"]
input_file = "csv_file/member_description_list_for_test_attributes.csv"
output_file = "csv_file/member_description_list_for_test_attributes_infered.csv"

add_user_attributes(
    input_file=input_file,
    output_file=output_file,
    column_name=column_name,
    new_column_names=new_column_names,
    delimiter=",",
    delimiter2=","
)