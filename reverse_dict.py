def reverse_dict(dic):
    out = {}
    for k,v in dic.items():
        out[v] = k
    keys = sorted(out.keys(),reverse=True)
    for k in keys:
        print(k,out[k])
    return out

if __name__ == '__main__':
    dic = {'Wangbing':1001,'Maling':1003,'Xulei':1004}
    result = reverse_dict(dic)
    print(result)