import csv
from sklearn.neural_network import MLPClassifier 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
c=0

arr=[]
with open('ml-100k/u1.test', 'rb') as csvfile:
    f=csv.reader(csvfile, delimiter='\n')

    for i in f:
        i=i[0]
        lines=i.split('\t')
        c=c+1
        lines=map(int,lines)
        '''
        if lines[2]==0 or lines[2]==1:
            lines[2]=1
        elif lines[2]==3:
            lines[2]=2
        else:
            lines[2]=3
        '''
        arr.append(lines)
        

genre=[]
print 'done0'
occupation=['administrator','artist','doctor','educator','engineer','entertainment','executive','healthcare','homemaker','lawyer','librarian','marketing','none','other','programmer','retired','salesman','scientist','student','technician','writer']
with open('ml-100k/u.item', 'rb') as csvfile:
    f=csv.reader(csvfile, delimiter='\n')

    for i in f:
        i=i[0]
        lines=i.split('|')
        lines[5]=(int)(lines[5])
        lines[6]=(int)(lines[6])
        lines[7]=(int)(lines[7])
        lines[8]=(int)(lines[8])
        lines[9]=(int)(lines[9])
        lines[10]=(int)(lines[10])
        lines[11]=(int)(lines[11])
        lines[12]=(int)(lines[12])
        lines[13]=(int)(lines[13])
        lines[14]=(int)(lines[14])
        lines[15]=(int)(lines[15])
        lines[16]=(int)(lines[16])
        lines[17]=(int)(lines[17])
        lines[18]=(int)(lines[18])
        lines[19]=(int)(lines[19])
        lines[20]=(int)(lines[20])
        lines[21]=(int)(lines[21])
        lines[22]=(int)(lines[22])
        
        
        #lines=map(int,lines)
        #print lines
        genre.append(lines)    
print 'done1'
print "Genre :\n",genre
userinfo=[]
with open('ml-100k/u.user', 'rb') as csvfile:
    f=csv.reader(csvfile, delimiter='\n')

    for i in f:
        i=i[0]
        lines=i.split('|')
        lines[0]=(int)(lines[0])
        lines[1]=(int)(lines[1])
        if lines[2]=='M':
            lines[2]=0
        elif lines[2]=='F':
            lines[2]=1
        lines[3]= occupation.index(lines[3])
        userinfo.append(lines)    
print "Userinfo :\n",userinfo
input1=[]
output=[]
for i in range(c):
    in1=[]
    in1.append(arr[i][0])
    ii=arr[i][0]-1
    iii=arr[i][1]-1
    age=userinfo[ii][1]
    gender=userinfo[ii][2]
    in1.append(age)
    in1.append(gender)
    occ=userinfo[ii][3]
    #in1.append(arr[i][3])
    in1.append(occ)
    output.append(arr[i][2])
    sum1=0
 
    for jj in range(5,23):
        sum1=sum1+genre[iii][jj]
    #in1.append(sum1)
    input1.append(in1)
print 'done2'
inputt=np.array(input1)
outputt=np.array(output)


print outputt

clf=IsolationForest()
print '1'
clf.fit(inputt,y=None)   
print '2'
filteredX=clf.predict(inputt)   
print '3'
c=0
for i in range(0,len(filteredX)):
   
    if filteredX[i]==-1:
        c=c+1
print c
print '4'
XX=[]
YY=[]
for i in range(0,len(filteredX)):
    if filteredX[i]==1:
        XX.append(inputt[i])
        YY.append(outputt[i])
print '5'

np.savetxt('YY5Nogenretest.txt',outputt,fmt='%s')
np.savetxt('XX5Nogenretest.txt',inputt,fmt='%s')


print 'com'

