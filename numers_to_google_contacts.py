import uuid
with open('numbers_ready_1.txt') as f:
    lines = f.readlines()
arr_of_unique_nums=[]
arr_of_nums=lines[0].split(',')
for num in arr_of_nums:
    if len(num)>5:
        full_num=f'+994{num}'
        print(full_num)
        if full_num not in arr_of_unique_nums:
            arr_of_unique_nums.append(full_num)

print(len(arr_of_nums))
print(len(arr_of_unique_nums))

for u_num in arr_of_unique_nums:
    name='Pasient '+str(uuid.uuid4())[0:12]
    line_=f'{name},,,,,,,,,,,,,,,,,,,,,,,,,,,,* myContacts ::: * starred,Mobile,{u_num}\n'
    print(line_)
    with open('contacts.csv', 'a') as fd:
        fd.write(line_)