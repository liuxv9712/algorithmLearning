class Solution:
    def twoSum(self, nums, target):
        if not nums:
            return None
        d = dict()
        for i, item in enumerate(nums):
            tmp = target - item
            if tmp in d:
                return [i, d[tmp]]
            d[item] = i
        return None

#How to use function?
if __name__ == '__main__':
    nums = [2, 7, 11, 45]
    target = 9
    solution = Solution()
    print(solution.twoSum( nums, target))
