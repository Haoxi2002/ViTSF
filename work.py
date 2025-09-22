with open('DLinear_A6000.txt', 'r', encoding='utf8') as f:
    l = []
    for line in f.readlines():
        if 'cost time:' in line:
            l.append(float(line.split('cost time: ')[1].split('\n')[0]))
    print(len(l))
    print("{:.3f} {:.3f} {:.3f} {:.3f} ".format(sum(l[:13])/13, sum(l[13:17])/4, sum(l[17:21])/4, sum(l[21:])/12))