"""
### utils.py
- contains some utility function

"""
def addProb(df,k,prob):
    df[k+"_prob0"]=prob[:,0]
    df[k+"_prob1"]=prob[:,1]
    return df

def ifnone(a,b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

fillMean=lambda x:x.fillna(x.mean())
def hardVoting(x):
    s=x.sum()
    ones=s
    zeros=4-ones
    if zeros>ones:
        res=0
    else:
        res=1
    return res
