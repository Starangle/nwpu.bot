import numpy as np
def lcs(str1,str2):
    len1=len(str1)
    len2=len(str2)
    dp = np.zeros([len1 + 1,len2 + 1],dtype=int)
    for i in range(len1+1):
        for j in range(len2+1):
            if i==0 or j==0:
                dp[i][j]=0
            elif str1[i-1]==str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len1][len2]

if __name__=='__main__':
    fr=open(r"data\BoP2017-DBQA.dev.txt","r",encoding='utf8')
    fw=open(r"data\resul3.txt","w+",encoding='utf8')
    for line in fr.readlines():
        words=line.split('\t')
        fw.write(str(lcs(words[1],words[2]))+'\n')
    fr.close()
    fw.close()