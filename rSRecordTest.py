import math

def RSME(records):
    #rresult = math.sqrt(sum([(rui-pui) ** 2 for u,i,rui,pui in records]))/ float(len(records))
    #return rresult
    return math.sqrt(\
        sum([(rui - pui) * (rui - pui) for u, i, rui, pui in records])\
        / float(len(records)))
def MAE(records):
    #mresult = math.sum([abs(rui-pui) for u,i,rui,pui in records])/ float(len(records))
    #return mresult
    return sum([abs(rui - pui) for u, i, rui, pui in records])\
        / float(len(records))
if __name__ == '__main__':
    records = [1,2,3,5]
    r1 = RSME(records)
    r2 = MAE(records)
    print(r1)
    print(r2)