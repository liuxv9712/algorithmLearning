n = input('Enter a number:')
print(n,'=',end = '')
i = 1
while n  != 1:
    while n % i == 0:
        n //= i
        if n!=1:
            print('{:d}'.format(i))
        else:
            print('{:d} *'.format(i),end = '')
    i+=1