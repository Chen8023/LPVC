import csv
import codecs
import numpy as np

#from pandas.tests.io.excel.test_xlrd import xlwt


def text_save(filename, data):
    file = open(filename,'a',encoding='gbk')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)

    file.write('\nAverage of all sequences:' + str(np.average(data)))

    file.close()

def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name,'w+','utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
