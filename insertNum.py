def insert(data,num):
    length=len(data)
    data.append(num)
    for i in range(length+1):
        if num<data[i]:
            for j in range(length,i,-1):
                data[j]=data[j-1]
            data[i]=num
            break

if __name__ == '__main__':
    data=[13,22,31,48,54,71,91,94]
    while True:
        try:
            num=int(input('Enter a new number:'))
            insert(data,num)
            print('The new sorted list is: ',data)
            continue
        except ValueError:
            print('Please enter a digit:')
