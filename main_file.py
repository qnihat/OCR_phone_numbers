import os
from detect_numbers import detect_num
from detect_symbols import detect_sym
from rabbit_producer import create_mq_queue
from number_arranger import arrabge_numbers
from save_nums_to_file import save_numbers_to_file
from rabbit_consumer import Start_rabbit_consumer
from config import SOURCE_IMAGES
from config import RABBIT_QUEUE_NAME

dir_ext_nums='extracted_numbers/'
dir_raw_images=SOURCE_IMAGES


create_mq_queue(RABBIT_QUEUE_NAME)

def extracted_nums_to_num_files():
    for file in os.listdir(dir_ext_nums):
        if '.jpg' in file:
            file=dir_ext_nums+file
            dict_of_=detect_sym(file)
            final_number=arrabge_numbers(dict_of_,file)
            print(final_number)
            save_numbers_to_file(final_number)

            #cv2.imshow('number', cv2.imread(file))
            #cv2.waitKey(0)

def extc_nums_from_images():
    for file in os.listdir(dir_raw_images):
        if '.jpg' in file:
            file=dir_raw_images+file
            detect_num(file)

extc_nums_from_images()
