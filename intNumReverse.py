def proc(n):
    xx=[]
    yy=0
    while True:
        if n!=0:
            xx.append(n%10)
        n//=10
        if n==0:
            break
    xx.sort()
    for i in xx:
        yy = yy*10+i
    return yy

print(proc(947821))