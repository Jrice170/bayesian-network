import numpy as np
import pandas as pd
import yaml
from pomegranate import *

#df = pd.read_csv('contact-lenses.csv')
df = pd.read_csv('hypothyroid.csv')

#print(df)


#N = df.index.size

## returns a list for each value
## map_value(df,'age') retuns list of values that ouccr
def map_value(df,atribute_str):

    dix = {}
    for i in df[atribute_str]:
        dix[i] = i


    return list(dix)

def Condition_prob(df,atribute1,value1,atribute2,value2,k_value,lambda1):

    Top = df.loc[(df[atribute1] == value1) & (df[atribute2]==value2),:].index.size
    Botton = df.loc[(df[atribute1] == value1),:].index.size

    return (Top + lambda1)/(Botton + k_value*lambda1)


#print(df.loc[(df['age']=='young') & (df['contact-lenses']=='none'),:])

def Create_bayesian_network(data_fram,lambdaa=1,title=''):
    N = data_fram.index.size
    DiscreatDistribution_list = []
    atribute_list = list(data_fram.columns)

    model = BayesianNetwork(title)

    Decision_atrbute_str = atribute_list[len(atribute_list) - 1]
    Dictionary1 = {}
    for each in map_value(data_fram,Decision_atrbute_str):
        K = len(map_value(data_fram,Decision_atrbute_str))
        Dictionary1[each] = ((data_fram.loc[data_fram[Decision_atrbute_str]==each,:]\
        .index.size)+(lambdaa))/(N+K*lambdaa)

    DiscreatDistribution_list.append(DiscreteDistribution(Dictionary1))



    for i in range(0,len(atribute_list)-1):
        list2 = []
        for k in map_value(data_fram,Decision_atrbute_str):
            for j in map_value(data_fram,atribute_list[i]):
                list2.append([k,j,Condition_prob(data_fram,Decision_atrbute_str,k,\
                atribute_list[i],j,\
                len(map_value(data_fram,atribute_list[i])),lambdaa)])

        G = ConditionalProbabilityTable(list2,[DiscreatDistribution_list[0]])

        DiscreatDistribution_list.append(G)

    model = BayesianNetwork(title)
    S1 = Node(DiscreatDistribution_list[0],name=Decision_atrbute_str)
    model.add_state(S1)


    ##model.add_states(DiscreatDistribution_list[0])
    ##model.add_edge(DiscreatDistribution_list[0],DiscreatDistribution_list[1])



    for i in range(1,len(DiscreatDistribution_list)):


        S = Node(DiscreatDistribution_list[i],name=atribute_list[i-1])

        model.add_state(S)

        model.add_edge(S1,S)


    model.bake()
    return model




new_model = Create_bayesian_network(df,1,"name of ......")


print("input: ",['negative',None,None,None,None,None,None,None,None,None])

print(new_model.predict([['negative',None,None,None,None,None,None,None,None,None]]))

print("input: ",['primary_hypothyroid',"M",None,None,None,None,None,None,None,'SVI'])
print(new_model.predict([['primary_hypothyroid',"M",None,None,None,None,None,None,None,'SVI']]))
