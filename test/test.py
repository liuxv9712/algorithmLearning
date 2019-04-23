import requests
response = requests.get('http://fund.sciencenet.cn/index.php/search/project?name=&person=&no=&company=&subject=&money1=&money2=&startTime=1999&endTime=2018&subcategory=&redract_url=&submit.x=77&submit.y=9&page=43139')
with open('./a.html','w',encoding='utf-8')as ff:
    ff.write(response.text)
print(response.text)