import sys

# n = int(sys.stdin.readline())
zero_array = [1,0]
one_array = [0,1]

def fibo(num):
    for x in range(2,num+1):
        zero_array.append(zero_array[x-2]+zero_array[x-1])
        one_array.append(one_array[x-2]+one_array[x-1])

n = 10
fibo(n)
print("%d %d"%(zero_array[n],one_array[n]))
