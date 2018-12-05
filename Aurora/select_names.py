def parseNL(path):
    f = open(path, 'r')
    names = []
    labels = []
    lines = f.readlines()
    for line in lines:
        if len(line.split()) == 1:
            name = line.split()
            names.append(name[0])
        if len(line.split()) == 2:
            [name, label] = line.split()
            names.append(name)
            labels.append(label)
    if len(labels) == 0:
        return names
    else:
        return names, labels

file_name_labels = '/home/ljm/NiuChuang/AuroraObjectData/Alllabel2003_38044.txt'
file_name_select = '/home/ljm/NiuChuang/AuroraObjectData/Alllabel2003_38044_arc.txt'
fs = open(file_name_select, 'w')

names, labels = parseNL(file_name_labels)

num_names = len(names)

for n in range(num_names):
    if labels[n] == '1':
        fs.write(names[n] + '\n')

pass