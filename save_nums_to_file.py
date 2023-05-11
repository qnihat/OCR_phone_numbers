

def save_numbers_to_file(number):
    if number is None:
        print('empty')
    else:
        with open("numbers.txt", "a") as myfile:
            myfile.write(number+' ,')
