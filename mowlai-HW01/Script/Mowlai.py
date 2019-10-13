# ===== PROBLEM1 =====
# Exercise 1 - Introduction - Say "Hello, World!" With Python

print("Hello, World!")

# Exercise 2 - Introduction - Python If-Else

n = int(input().strip())
if (n%2 == 1) :
    print("Weird")
elif (n%2 == 0) and (n in range(6,21)):
    print("Weird")
else:
    print("Not Weird")

# Exercise 3 - Introduction - Arithmetic Operators

a = int(input())
b = int(input())
if (1 <= a <= 10**10) and (1<=b<=10**10):
    print(a+b)
    print(a-b)
    print(a*b)

# Exercise 4 - Introduction - Python: Division

a = int(input())
b = int(input())
print(a//b)
print(a/b)


# Exercise 5 - Introduction - Loops

n = int(input())
if 1<=n<=20:
    for i in range(0,n):
        print(i**2)

# Exercise 6 - Introduction - Write a function

def is_leap(year):
    leap = False
    if (year%4 == 0) and (year%100!=0) and(1900<=year):
        leap = True
    elif (year%100 == 0)and(year%400==0):
        leap = True
    else:
        leap = False
    return leap


# Exercise 7 - Introduction - Print Function

n = int(input())
for i in range(1,n+1):
    print(i,end="")


# Exercise 8 - Basic data types - List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    list=[]
    for i in range(0,x+1):
        for j in range(0,y+1):
            for k in range(0,z+1):
                if i+j+k!=n:
                    list.append([i,j,k])
    print(list)


# Exercise 9 - Basic data types - Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int,input().strip().split()))[:n] 
a = max(arr)
while max(arr) == a:
    arr.remove(max(arr))

print (max(arr))


# Exercise 11 - Basic data types - Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    N=0
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    for names in student_marks:
        if names == query_name:
            for marks in student_marks[names]:
                N += marks
            average= N / len(student_marks[names])
    print("%.2f"%average)

# Exercise 12 - Basic data types - Lists

if __name__ == '__main__':
    N = int(input())
    l=[]
    for i in range(N):
        string = input().split(" ")
        if string[0]=="insert":
            l.insert(int(string[1]),int(string[2]))
        if string[0]== "print":
            print(l)
        if string[0]=="remove":
            l.remove(int(string[1]))
        if string[0]=="append":
            l.append(int(string[1]))
        if string[0]=="sort":
            l.sort()
        if string[0]=="pop":
            l.pop()
        if string[0]=="reverse":
            l.reverse()


# Exercise 13 - Basic data types - Tuples

n = int(input())
integer_list = map(int, input().split())
t= tuple(integer_list)
print(hash(t))

# Exercise 14 - Strings - sWAP cASE

def swap_case(s):
    a = ""
    for i in s:
        if i.islower()==True:
            a += (i.upper())
        elif i.isupper()==True:
            a += (i.lower())
        else:
            a += i
    return(a)
  
if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

# Exercise 15 - Strings - String Split and Join

def split_and_join(line):
    # write your code here
    string = (line)
    string = string.split(" ")
    string = "-".join(string)
    return(string)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)





# Exercise 16 - Strings - What's Your Name?

def print_full_name(a, b):
    s = []
    s.append(a)
    s.append(b)
    
    print("Hello"+" "+ s[0]+" "+s[1]+ "! You just delved into python.")
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

# Exercise 17 - Strings - Mutations

def mutate_string(string, position, character):
    s = string
    i = position
    ch =c haracter
    s_new=[]
    s_new=[s]
    s_new=s[:int(i)]+str(ch)+s[int(i+1):]
    return(s_new)
if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

# Exercise 18 - Strings - Find a string

def count_substring(string, sub_string):
    c = 0
    count = 0
    for i in range(len(string)-1):
        for j in range (len(sub_string)-1):
            if string[i]==sub_string[j]:
                if string[i:i+len(sub_string)]==sub_string:
                    count+=1

    return(count)
if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

# Exercise 19 - Strings - String Validators
#with some help from disscusion 
if __name__ == '__main__':
    s = input()

result = ["False", "False", "False", "False", "False"]
for i in s:
    if i.isalnum():
        result[0] = "True"
    if i.isalpha():
        result[1] = "True"
    if i.isdigit():
        result[2] = "True"
    if i.islower():
        result[3] = "True"
    if i.isupper():
        result[4] = "True"
for i in result:
    print(i)

# Exercise 20 - Strings - Text Alignment

 
thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# Exercise 21 - Strings - Text Wrap

import textwrap
def wrap(string, max_width):
    e=""
    for i in range(0,len(string),max_width):
        e+=(string[i:max_width+i])+"\n"
    return(e)


if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# Exercise 22 - Strings - Designer Door Mat

def door(number1, number2):
    number1 = int(number1)
    number2 = int(number2)
    for i in range(1, number1, 2): 
        print ((".|." * i ).center(number2, "-"))
    print ("WELCOME".center(number2, "-"))
    for i in range(1, number1, 2): 
        print ((".|." * (number1-1-i) ).center(number2, '-'))
door(*input().split())




# Exercise 29 - Sets - No Idea!

import sys
inp = sys.stdin.readlines()
happiness = 0
for i in range(len(inp)):
    inp[i] = inp[i].strip("\n")   
listarray = list(inp[1].split(" "))
A = set(inp[2].split(" "))
B = set(inp[3].split(" "))
print (sum([(i in A) - (i in B) for i in listarray]))

# Exercise 30 - Sets - Symmetric Difference

import sys
inplines = sys.stdin.readlines()
for i in range(len(inplines)):
    inplines[i] = inplines[i].strip("\n")
m = set(inplines[1].strip().split())
n = set(inplines[3].strip().split())
uni = m.symmetric_difference(n)

uni1 = sorted(list(map(int, list(uni))))
for i in uni1:
    print(i)

# Exercise 31 - Sets - Set .add()

import sys
inp = sys.stdin.readlines()
for i in range(len(inp)):
    inp[i] = inp[i].strip("\n")
inp.pop(0)  
print(len(set(inp)))

# Exercise 32 - Sets - Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
import sys 
lines = sys.stdin.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].strip("\n")
lines.pop(0)
for j in range(len(lines)):
    lines[j] = lines[j].split()
    if lines[j][0] == 'pop':
        s.pop()
    elif lines[j][0] == 'remove':
        s.remove(int(lines[j][1]))
    elif lines[j][0] == 'discard':
        s.discard(int(lines[j][1]))
    
    
print(sum(s))

# Exercise 33 - Sets - Set .union() Operation

import sys
lines = sys.stdin.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].strip("\n")
A = set(lines[1].split(" "))
B = set(lines[3].split(" "))
uni = A.union(B)
print(len(uni),end="")

# Exercise 34 - Sets - Set .intersection() Operation

import sys
lines = sys.stdin.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].strip("\n")
A = set(lines[1].split(" "))
B = set(lines[3].split(" "))
uni = A.intersection(B)
print(len(uni),end="")

# Exercise 35 - Sets - Set .difference() Operation

import sys
lines = sys.stdin.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].strip("\n")
A = set(lines[1].split(" "))
B = set(lines[3].split(" "))
uni = A.difference(B)
print(len(uni),end="")

# Exercise 36 - Sets - Set .symmetric_difference() Operation

import sys
lines = sys.stdin.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].strip("\n")
A = set(lines[1].split(" "))
B = set(lines[3].split(" "))
uni = A.symmetric_difference(B)
print(len(uni),end="")

# Exercise 37 - Sets - Set Mutations
#Done with a little help from internet
N = int(input())
elem = list(map(int, input().split()))
s = set(elem)
otherlines = int(input()) * 2
lis = []
liscom = []
lisset = []
for i in range(otherlines):
    lis.append(input().split())

for i in range(0,otherlines,2):
    liscom.append(lis[i])
for i in range(1,otherlines,2):
    lisset.append(lis[i])   
for i in range(len(lisset)):  
    lisset[i] = set(map(int, lisset[i]))
for i in range(len(liscom)):
    if liscom[i][0] == "intersection_update":
        s.intersection_update(lisset[i])
    if liscom[i][0] == "update":
        s.update(lisset[i])
    if liscom[i][0] == "symmetric_difference_update":
        s.symmetric_difference_update(lisset[i])
    if liscom[i][0] == "difference_update":
        s.difference_update(lisset[i])            
               
    
print(sum(s))

# Exercise 38 - Sets - The Captain's Room

n = int(input())
rooms = list(map(int,input().split(" ")))
setrooms = set(rooms)
Captan = ((sum(setrooms)*n)-(sum(rooms)))//(n-1)
print(Captan)


# Exercise 39 - Sets - Check Subset
#Actually didn't know anything about issubset() and searched and learned how to use it.
list = []
for _ in range(int(input())):
        nA = int(input())
        setA = set(map(int, input().split()))
        nB = int(input())
        setB = set(map(int, input().split()))
        list.append(setA.issubset(setB))
for i in list:
        print(i)

# Exercise 40 - Sets - Check Strict Superset
#Actually didn't know anything about issuperset() and searched and learned how to use it.
A = set(map(int, input().split()))
N = int(input())
sli = []
for i in range(N):
    sets = set(map(int, input().split()))
    sli.append(sets)
result = True
for i in sli:
    if not A.issuperset(i):
        result = False
print(result)

# Exercise 41 - Collections - collections.Counter()


from collections import Counter
nsi = int(input())
sisho = Counter(map(int,input().split(" ")))
ncu = int(input())
l = []
for i in range(ncu):
    cusi,price = map(int,input().split(" "))
    if sisho[cusi]:
        l.append(price)
        sisho[cusi] -= 1
print(sum(l))

# Exercise 42 - Collections - DefaultDict Tutorial

from collections import defaultdict
d = defaultdict(list)
li=[]

n, m = map(int,input().split())

for i in range(0,n):
    d[input()].append(i+1) 

for i in range(0,m):
    li=li+[input()]  

for i in li: 
    if i in d:
        print (" ".join( map(str,d[i]) ))
    else:
        print(-1)

# Exercise 43 - Collections - Collections.namedtuple()
#with some help
stu= int(input())
marks=input().split().index("MARKS")
print (sum([int(input().split()[marks]) for _ in range(stu)]) / stu)

# Exercise 44 - Collections - Collections.OrderedDict()
#with some help
from collections import OrderedDict
ordered_dictionary = OrderedDict()
for _ in range(int(input())):
    item, price = input().rsplit(' ', 1)
    ordered_dictionary[item] = ordered_dictionary.get(item, 0) + int(price)
[print(item, ordered_dictionary[item]) for item in ordered_dictionary]

# Exercise 45 - Collections - Word Order

from collections import OrderedDict
dict = OrderedDict()
for i in range(int(input())):
    key = input()
    if not key in dict.keys():
        dict.update({key : 1})
        continue
    dict[key] += 1

print(len(dict.keys()))
print(*dict.values())

# Exercise 46 - Collections - Collections.deque()

from collections import deque
d = deque()
for _ in range(int(input())):
    inp = input().split()
    getattr(d, inp[0])(*[inp[1]] if len(inp) > 1 else [])
print(*[item for item in d])

# Exercise 48 - Collections - Piling Up!
#with some help
for t in range(int(input())):
    input()
    l = [int(i) for i in input().split()]
    min_list = l.index(min(l))
    left = l[:min_list]
    right = l[min_list+1:]
    if left == sorted(left,reverse=True) and right == sorted(right):
        print("Yes")
    else:
        print("No")

# Exercise 49 - Date time - Calendar Module
#with some help
import calendar
m,d,y=map(int,input().split())
print((calendar.day_name[calendar.weekday(y,m,d)]).upper())

# Exercise 50 - Date time - Time Delta
#with some help
from datetime import datetime as dt

fmt = '%a %d %b %Y %H:%M:%S %z'
for _ in range(int(input())):
    time1 = dt.strptime(input(), fmt)
    time2 = dt.strptime(input(), fmt)
    print(int(abs((time1 - time2).total_seconds())))

# Exercise 51 - Exceptions -

for i in range(int(input())):
    try:
        a,b = map(int,input().split()) 
        division_result = a // b
        print(division_result)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)

# Exercise 52 - Built-ins - Zipped!
#with some help
n, x = map(int,input().split())
all_scores = [map(float, input().split()) for _ in range(x)]

for student, score in zip(range(1, n+1), zip(*all_scores)):
    print (sum(score)/x)

# Exercise 53 - Built-ins - Input()

inp = input().split()
x = int(inp[0])
print(eval(input()) == int(inp[1]))

# Exercise 54 - Built-ins - python evaluation

eval(input())


# Exercise 56 - Regex - Detect Floating Point Number
#Actually i got some help for this secsion from one of my colleages and totally understood.
import re
n = int(input())
inp = [input().split() for i in range(n)]
pt = re.compile(r"(^[+-]?\d{1,}\.\d{1,}$)|(^[+-]?\.\d{1,}$)")
for i in range(n):
    if pt.search(inp[i][0]):
        if float(inp[i][0]):
            print("True")     
    else:
        print("False")

# Exercise 58 - Regex - Group(), Groups() & Groupdict()

import re
s = input().strip()
res = re.search(r'([a-zA-Z0-9])\1+', s)
if res:
    print(res.group(1))
else:
    print(-1)

# Exercise 59 - Regex - Re.findall() & Re.finditer()

import re
inp = input()
res = re.findall(r'(?=[^aeiouAEIOU]([AEIOUaeiou]{2,})[^aeiouAEIOU])',inp)
print('\n'.join(res or ['-1']))


# Exercise 60 - Regex - Re.start() & Re.end()
# Exercise 61 - Regex - Regex Substitution

import re
N = int(input())
lines =[input() for i in range(N)]
for i in range(N):
    lines[i] = re.sub(r" \&\&(?= )"," and",lines[i])
for i in range(N):
    lines[i] = re.sub(r" \|\|(?= )"," or",lines[i])    

for i in range(N):
    print(lines[i])

# Exercise 62 - Regex - Validating Roman Numerals
# Exercise 63 - Regex - Validating phone numbers

import re
N = int(input())
phones = [input() for i in range(N)]
ptn = re.compile(r"^[789]\d{9}$")
for i in range(N):
    if re.search(ptn,phones[i]):
        print("YES")
    else:
        print("NO")

# Exercise 64 - Regex - Validating and Parsing Email Addresses
# Exercise 65 - Regex - Hex Color Code

import re
N = int(input())
CODE = [input().split("\n") for i in range(N)]

for i in range(len(CODE)):
     res = re.findall(r":?.(#[0-9a-fA-F]{3,6})",CODE[i][0])
     if res:
          for i in range(len(res)):
              print(res[i])

# Exercise 66 - Regex - HTML Parser - Part 1
# Exercise 67 - Regex - HTML Parser - Part 2
# Exercise 68 - Regex - Detect HTML Tags, Attributes and Attribute Values
# Exercise 69 - Regex - Validating UID
# Exercise 70 - Regex - Validating Credit Card Numbers
# Exercise 71 - Regex - Validating Postal Codes
# Exercise 72 - Regex - Matrix Script
# Exercise 73 - Xml - XML 1 - Find the Score

def get_attr_number(node):
    # your code goes here
    return sum(len(i.attrib) for i in root.iter())


# Exercise 74 - Xml - XML 2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)   


# Exercise 75 - Closures and decorators - Standardize Mobile Number Using Decorators

import re
def wrapper(f):
    def fun(l):
        pat = re.compile(r'(0|91|\+91)?(\d{5})(\d{5})')
        l = [re.sub(pat, r"+91 \2 \3", x) for x in l]
        f(l)        
    return fun


# Exercise 76 - Closures and decorators - Decorators 2 - Name Directory
#with some help

def person_lister(f):
    def inner(people):
        # complete the function
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner


# Exercise 77 - Numpy - Arrays


def arrays(arr):
    # complete this function
    # use numpy.array
    a= arr[::-1]
    b= numpy.array(a,float)
    return b


# Exercise 78 - Numpy - Shape and Reshape

import numpy
ar = list(map(int,input().rstrip().split()))
arr = numpy.array(ar)
#print(arr)
print(numpy.reshape(arr,(3,3)))


# Exercise 79 - Numpy - Transpose and Flatten

import numpy
import sys

N,M = list(map(int,input().strip().split()))
array = []
for i in range(N):
     inp = input().split()
     array.append(inp) 
for i in range(len(array)):
    array[i] = list(map(int,array[i]))

     
array = numpy.array(array)
print (numpy.transpose(array))
print (array.flatten())

# Exercise 80 - Numpy - Concatenate
#with some help but toatally understood
import numpy

N, M, P = list(map(int,input().strip().split()))
l1 = []
l2 = []
for i in range(N):
    inp = list(map(int, input().split()))
    l1.append(inp)
for i in range(N+1, N+M+1):
    inp = list(map(int, input().split()))
    l2.append(inp)

arr1 = numpy.array(l1)
arr2 = numpy.array(l2)
final = numpy.concatenate((arr1,arr2), axis=0)
print(final, end="")


# Exercise 81 - Numpy - Zeros and Ones

import numpy

A = tuple(map(int, input().strip().split()))
zero = numpy.zeros(A, dtype=int)
ones = numpy.ones(A, dtype=int)
print(zero)
print(ones)

# Exercise 82 - Numpy - Eye and Identity
#with some help but toatally understood
import numpy
N, M = map(int, input().split())
numpy.set_printoptions(sign=' ')
print(numpy.eye(N,M))


# Exercise 83 - Numpy - Array Mathematics

import numpy

N, M = map(int, input().split())
a = numpy.array([list(map(int, input().split())) for i in range(N)], int)
b = numpy.array([list(map(int, input().split())) for i in range(N)], int)
#print(a,b)
print(a+b,"\n",a-b,"\n",a*b,"\n",a//b,"\n",a%b,"\n",a**b)



# Exercise 84 - Numpy - Floor, Ceil and Rint

import numpy
inp = numpy.array(list(map(float, input().split())))
numpy.set_printoptions(sign=" ")
print(numpy.floor(inp))
print(numpy.ceil(inp))
print(numpy.rint(inp))


# Exercise 85 - Numpy - Sum and Prod

import numpy
N, M = map(int, input().split())
array = numpy.array([list(map(int,input().split())) for i in range(N)])
print(numpy.prod(numpy.sum(array , axis=0),axis =0))


# Exercise 86 - Numpy - Min and Max

import numpy
N, M = map(int, input().split())

arr = numpy.array([list(map(int, input().split())) for i in range(N)])

arr1 = numpy.min(arr, axis=1)
print(max(array1))


# Exercise 88 - Numpy - Dot and Cross

import numpy

N = int(input())
inp1 = [list(map(int, input().split())) for i in range(N)]
A = numpy.array(inp1)
inp2 = [list(map(int, input().split())) for i in range(N)]
B = numpy.array(inp2)
print(numpy.dot(A, B))


# Exercise 89 - Numpy - Inner and Outer

import numpy

A = numpy.array(list(map(int, input().split())), int)
B = numpy.array(list(map(int, input().split())), int)
print(numpy.inner(A,B))
print(numpy.outer(A,B))

# Exercise 90 - Numpy - Polynomials

import numpy
l1 = list(map(float, input().split()))
A = int(input())
print(numpy.polyval(l1, A))

# Exercise 91 - Numpy - Linear Algebra

import numpy

A = int(input())
arr = numpy.array([input().split() for i in range(A)],float)
numpy.set_printoptions(legacy='1.13')
print(numpy.linalg.det(arr))



# ===== PROBLEM2 =====


# Exercise 92 - Challenges - Birthday Cake Candles

#!/bin/python3

import math
import os
import random
import re
import sys
from collections import Counter
# Complete the birthdayCakeCandles function below.
def birthdayCakeCandles(ar):
    a = list(Counter(ar).items())
    a.sort()
    

    return a[-1][1]
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()

# Exercise 93 - Challenges - Kangaroo

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if x2>=x1 and v2>v1 :
        return("NO")
    elif (v1!=v2) and (abs(x2-x1))%abs((v1-v2)) == 0:
        return("YES")
    else:
        return("NO")

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Exercise 94 - Challenges - Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    shared = 5
    like = 0
    cumu = 0
    for i in range(n):
        like = shared//2
        cumu = cumu + like
        shared = like*3
    return(cumu)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Exercise 95 - Challenges - Recursive Digit Sum

n, k = map(int, input().split())
x = n * k % 9
print(x if x else 9)

# Exercise 96 - Challenges - Insertion Sort - Part 1

#!/bin/python3

import sys

# Complete the insertionSort1 function below.

def insertionSort1(n, arr):
    a = arr[-1]
    i = n-2
    
    while (a < arr[i]) and (i >= 0):
        arr[i+1] = arr[i]
        print(' '.join(map(str, arr)))
        i -= 1
        
    arr[i+1] = a
    print(' '.join(map(str, arr)))
    
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    insertionSort1(n, arr)


# Exercise 97 - Challenges - Insertion Sort - Part 2
#Done with some help

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n,arr):
    for i in range(1,n):
        c =arr[i]
        j=i-1
        while j>=0 and arr[j]>c:
            arr[j+1]=arr[j]
            j=j-1
        arr[j+1]=c
        print(' '.join(str(i) for i in arr))
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    insertionSort2(n, arr)



