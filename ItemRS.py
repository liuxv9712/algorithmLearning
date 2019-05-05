import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#生成一个用户-项目-评分矩阵
df = pd.DataFrame({'U1':[2,None,1,None,3],'U2':[None,3,None,4,None],'U3':[4,None,5,4,None],'U4':[None,3,None,4,None],'U5':[5,None,4,None,5]})
df.index = ['S1','S2','S3','S4','S5']
print(df)


def get_sim(ratings,target_user,target_item,k=2):
    centered_ratings = ratings.apply(lambda x:x-x.mean(),axis=1)#mean()函数功能：求取均值 axis 不设置值，对 m*n 个数求均值，返回一个实数;axis = 0：压缩行，对各列求均值，返回 1* n 矩阵;axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    print(centered_ratings)
    csim_list = []
    #求相似度
    for i in centered_ratings.index:
        csim_list.append(cosine_similarity(np.nan_to_num(centered_ratings.loc[i,:].values).reshape(1,-1),np.nan_to_num(centered_ratings.loc[target_item,:]).reshape(1,-1)).item())
    print(csim_list)
    #列出其他所有的相似度和评分
    new_ratings = pd.DataFrame({'similarity':csim_list,'rating':ratings[target_user]},index=ratings.index)
    print(new_ratings)
    #删除为0的行
    top = new_ratings.dropna().sort_values('similarity',ascending=False)[:k].copy()
    print(top)
    top['multiple'] = top['rating']*top['similarity']
    print(top['multiple'])
    result = top['multiple'].sum()/top['similarity'].sum()
    print(result)
    return result

if __name__ == '__main__':
    get_sim(df,'U3','S5',2)

