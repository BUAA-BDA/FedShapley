import numpy as np

decay = 0.8
agentNum = 5


def normal(inList):
        symbolF = False
        for i in inList:
                if i < 0:
                        symbolF = True
        if symbolF:
                for i in range(len(inList)):
                        inList[i] = 0.1 / 5
        return inList


def get(FileName):

        with open("./output_"+FileName+"_MR.txt", "r") as fin:
                res = np.zeros(agentNum)

                print(FileName+" ", end="")
                lines = fin.readlines()
                index = 0
                for line in lines:
                        line = line.strip('\n')
                        line = line.replace("HISTORY ", "")
                        if len(line) < 2:
                                continue
                        if line[0] != '[':
                                continue
                        if line[-1] != ']':
                                continue
                        if line.find(',') == -1:
                                continue

                        w = eval(line)
                        sum = 1e-7  # zero
                        index = index + 1
                        # normal 处理小于0 的情况
                        w = normal(w)
                        for i in w:
                                sum = sum + i
                        mid = []
                        for i in w:
                                x = i / sum
                                mid.append(x * (decay**index))
                        res = np.add(res, mid)
                for o in range(5):
                        print(res[o], end=" ")
                print("")


get("Same")
get("Weight")
get("Mix")
get("NoiseX")
get("NoiseY")
