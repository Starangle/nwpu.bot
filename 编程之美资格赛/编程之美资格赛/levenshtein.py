import numpy as np
import codecs
'''效果非常差'''
def  levenshtein(str1,str2):
    len1 = len(str1)
    len2 = len(str2)
    dp = np.zeros([len1 + 1,len2 + 1],dtype=int)
    for i in range(0,len1 + 1):
        dp[i][0] = i
    for i in range(0,len2 + 1):
        dp[0][i] = i
    for i in range(1,len1 + 1):
        for j in range(1,len2 + 1):
            dp[i][j] = min(dp[i - 1][j - 1],dp[i - 1][j],dp[i][j - 1]) + 1
            if str1[i-1] == str2[j-1]:
                dp[i][j] == min(dp[i][j],dp[i - 1][j - 1])
    return dp[len1][len2]
