'''
假设举办一个聚会。你为每个进入聚会的人分配一个唯一的1～100之间的号码。现在聚会结束了，你宣布了一个消息。

“会从1～200之间获取一个随机数。如果有两个人的号码之和与这个数字相等，就会奖励这两个人。”

现在了解到，已经向x个人分配了号码。如何确定能否给其中两个人奖励呢？

通过上述语句，将排序后的数字赋值给变量。现在使用两个指针：一个指向开始处（前指针）；另一个指向结束处（后指针）。

检查两指针所指变量的和。如果和小于给定值（意味着当前的总和小于所需的总和），前指针向前移动一步并再次检查；

如果和大于给定值（意味着当前的总和大于所需的总和），将后指针向后移动一步并再次检查。

无论在任何位置，如果当前的总和等于要求的总和，便可以说，存在两个人的数值之和与给定的数值相同，需要给予奖励；

如果两个指针相交并且仍然没有达到所需的总和，则可以声称并不存在两个人的数值之和与给定的数值相等。
'''


def isPrizeGiven(numberList,sumSelector):
    sumOfTwo = sumSelector
    i = 0
    j = len(numberList) - 1
    if (i >= j):
        return False
    while (i <= j):
        currentSum = numberList[i] + numberList[j]
        print(i, j, currentSum)
        if (currentSum == sumOfTwo):
            return True
        if (currentSum > sumOfTwo):
            j = j - 1
        else:
            i = i + 1
    return False

if __name__ == '__main__':
    # 首先，从列表中获取输入数据。
    numberList = list([43, 23, 1, 67, 54, 2, 34, 56, 23, 65, 12, 9, 87, 4, 33])
    # 为了解决这个问题，必须先对数据进行排序。在Python列表中排序很简单，可以通过sort函数来完成。
    numberList = numberList.sort()
    # 选择一个随机数并将其存储在一个变量中
    #sumSelector = raw_input()
    sumSelector = input()
    result=isPrizeGiven(numberList, sumSelector)
    print(result)