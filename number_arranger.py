import re

def processing_number_txt(txt_num):
    match = re.search(r'[5-7-9][0-1-5-7-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]', txt_num)
    if match:
        processed_number = match.group()
        #print('FINAL NUMBER: ',processed_number)
        return processed_number

def arrabge_numbers(dict_data,file_path):
    list_of_dict_keys = list(dict_data.keys())
    list_of_dict_keys_sorted = sorted(list_of_dict_keys)
    num_list =''
    for el in list_of_dict_keys_sorted:
        # print('numbver: ',dict_of_[el])
        num_list+=dict_data[el]
    #print('final num: ', num_list)
    return processing_number_txt(num_list)
