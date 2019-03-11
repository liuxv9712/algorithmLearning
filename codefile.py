import math


def isprime(x):
    if x == 0 and x == 1:
        return False
    k = int(math.sqrt(x))
    for i in list(range(2, k + 1)):
        if x % i == 0:
            return False
    return True


def ismonisen(x):
    if isprime(x) and isprime(2 ** x - 1):
        return True
    else:
        return False


if __name__ == '__main__':
    num_list = [2, 7, 11, 13, 21]
    result_list = []
    for num in num_list:
        if ismonisen(num):
            temp = 2 ** num - 1
            print(temp,end=' ')
            result_list.append(str(temp)+'')
    with open('myf2.out', 'w') as fp:
        fp.writelines(result_list)
        fp.write('\n My exam number is : 1223123')
