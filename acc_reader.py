import os 
# read 0_class_acc.txt contents 
def read_acc(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines

# use the function read_acc to read the contents of 0_class_acc.txt
lines = read_acc('0_class_acc.txt')

# delete all lines with text in them
for line in lines:
    if line.strip():
        lines.remove(line)
# convert all values in list to float and compute the average
#print(lines)
print(len(lines))
acc_list = [float(line) for line in lines]
# get max and min values of the list
max_acc = max(acc_list)
min_acc = min(acc_list)
print("max", max_acc)
print("min", min_acc)
acc_avg = sum(acc_list)/len(acc_list)
print("avg", acc_avg)