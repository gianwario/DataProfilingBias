import re
import json


def transform_rfds(file_path):
    # Initialize the rfd list
    rfd = []
    
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Process each line to format it into the desired structure
    for line in lines:
        # Split the line into LHS and RHS
        lhs, rhs = line.strip().split('->')
        # Split LHS into a list of attributes and clean up any extra spaces
        lhs_list = [attr.strip() for attr in lhs.split(',')]
        # Strip any extra spaces from RHS and convert it to a list
        rhs_list = [rhs.strip()]
        # Append the formatted dictionary to the rfd list
        rfd.append({'LHS': lhs_list, 'RHS': rhs_list})
    
    return rfd

def filter_rfds(rfd_list):
    filtered_rfds = []
    for rfd in rfd_list:
        # Check if 'race' or 'sex' is in LHS and 'income' in RHS
        lhs_contains_race_or_sex = any(attr in ['race', 'sex'] for attr in rfd['LHS'])
        rhs_contains_income = any(attr in ['income'] for attr in rfd['RHS'])
        
        if lhs_contains_race_or_sex and rhs_contains_income:
            filtered_rfds.append(rfd)
    
    return filtered_rfds

def save_rfds_to_json(rfd_list, file_path):
    with open(file_path, 'w') as file:
        json.dump(rfd_list, file, indent=4)



if __name__ == "__main__":
    file_path = 'rfds/adult_1_0.1.txt'
    rfd_result = transform_rfds(file_path)
    rfd_result = filter_rfds(rfd_result)
    file_path = 'biased_processed_rfds/adult_1_0.1.json'
    save_rfds_to_json(rfd_result, file_path)

    file_path = 'rfds/adult_1_0.2.txt'
    rfd_result = transform_rfds(file_path)
    rfd_result = filter_rfds(rfd_result)
    file_path = 'biased_processed_rfds/adult_1_0.2.json'
    save_rfds_to_json(rfd_result, file_path)

    file_path = 'rfds/adult_2_0.1.txt'
    rfd_result = transform_rfds(file_path)
    rfd_result = filter_rfds(rfd_result)
    file_path = 'biased_processed_rfds/adult_2_0.1.json'
    save_rfds_to_json(rfd_result, file_path)

    file_path = 'rfds/adult_2_0.2.txt'
    rfd_result = transform_rfds(file_path)
    rfd_result = filter_rfds(rfd_result)
    file_path = 'biased_processed_rfds/adult_2_0.2.json'
    save_rfds_to_json(rfd_result, file_path)

    file_path = 'rfds/adult_3_0.1.txt'
    rfd_result = transform_rfds(file_path)
    rfd_result = filter_rfds(rfd_result)
    file_path = 'biased_processed_rfds/adult_3_0.1.json'
    save_rfds_to_json(rfd_result, file_path)


    file_path = 'rfds/adult_3_0.2.txt'
    rfd_result = transform_rfds(file_path)
    rfd_result = filter_rfds(rfd_result)
    file_path = 'biased_processed_rfds/adult_3_0.2.json'
    save_rfds_to_json(rfd_result, file_path)
