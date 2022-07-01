import numpy as np


fw1 = open("第1份结果-句子.txt", 'w', encoding = 'utf-8')
fw2 = open("第1份结果-实体及关系-有扰动.txt", 'w', encoding = 'utf-8')


def ttag(lab_chunks, trueRel, n, i, strstr):
    # print("------第"+ str(i) + "组" + strstr + "结果------", file=fw2)
    # print(lab_chunks, file=fw2)
    # print(trueRel, file=fw2)
    fw2.write("------第"+ str(i) + "组" + strstr + "结果------" + '\n')
    fw2.write(str(lab_chunks))
    fw2.write('\n')
    fw2.write(str(trueRel))
    fw2.write('\n')
    
   
   
    """
    {('SUB', 12, 12), ('SEN', 0, 0), ('DAD', 15, 16), ('SEN', 32, 32), ('Analysis_and_Inference_of_results', 25, 25), 
    ('COG', 43, 44), ('DAD', 18, 18), ('SUB_A', 30, 30), ('had_Member', 35, 35)}
    其中，('DAD', 15, 16)====('论元的标签',组成论元的第一个单词在整句话中的id,组成论元的最后一个单词在整句话中的id)
    """
        # print(trueRel, file = fw2)
    """
    [(12, 'SUB_rs', 30), (25, 'Analysis_and_Inference_of_results_rs', 44), (35, 'had_Member_rs', 44)]
    其中，(12, 'SUB_rs', 30)===(组成论元1的最后一个单词在整句话中的id,'预测出来的两个论元之间的关系',组成论元2的最后一个单词在整句话中的id)
    """


def old_data(lists):
   
    lists = lists.tolist() # 将数组转换成列表
    # print(len(lists))
    # print("===============================================")
    # print(lists)  # 没有问题
   
    stens = []
    for j in range(len(lists)):  # lists中放着多个列表，每个列表lists[j]由一句话组成
       
        if '#doc' in lists[j][0]:
            lists[j].insert(0, '-1')
            # stens.append(lists[j][1])
        # print(lists[j])  # 没有问题
        stens.append(lists[j][1])
    # print(len(stens))
    # print("===============================================")
    # print(stens)  # 没有问题，训练集中所有的有效字符
    
    st = ""
    for i in range(len(stens)):  # range(1, len(sten))
        
        if i == 0:  # 为了让i从1开始
            continue
        if type(stens[i-1]) == float:
            st = st + str(stens[i-1]) + ' '
        else:
            st = st + stens[i-1] + ' '
        if stens[i-1] == '.' and '#doc' in stens[i]:
            fw1.write(st.strip() + '\n')
            # fw1.write()
            st = ""
    fw1.write(st.strip() + ' ' + stens[-1])
    

#输出格式：单词 实体类列 最后一个词的位置 关系类型（标在前一个词上）目标词的位置
#每句话一个数组，一篇文章再用一个大组包起来








