#import necessary modules
import csv

with open('my_csv.csv', mode='w') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #way to write to csv file
    for i in range(150):
        writer.writerow(['25','Private','226802','11th','7','Never-married','Machine-op-inspct','Own-child','Black','Male','0','0','40','United-States','<=50K'])
        
    