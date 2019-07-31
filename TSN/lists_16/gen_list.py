# coding:utf-8

for i in range(16):
    with open("list.txt.{}".format(i), "r")as txt_f:
        data = txt_f.readlines()
    with open("list.txt.new{}".format(i), "w")as txt_f:
        for item in data:
            item = item.replace("datasets", "anet_flow")
            txt_f.write(item)
