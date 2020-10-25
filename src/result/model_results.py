
import numpy as np

def confirmValidate():
    confirm = input("\n\nDo you want to [v]validate model or [e]exit: ?")
    if confirm != 'v' and confirm != 'e':
        print("\n\nInvalid Option. Please Enter a Valid Option.")
        confirmValidate() 
    print (confirm)
    return confirm

def showSummary(result,validationData):
    printStats(getStats(validationData,result))

def showSummaryClassification(result,validationData):
    printStatsClassification(getStats(validationData,result))

def getStats(real,predicted):
    numClasses = real.shape[1]
    stats = []
    for i in range(0,numClasses):
        stats.append([])
        for j in range(0,numClasses):
            stats[i].append(0)
    for i in range(0,len(predicted)):
        p = np.argmax(predicted[i])
        r = np.argmax(real[i])
        stats[p][r]+=1
    metrics = {}
    for i in range(0,numClasses):
        tp = stats[i][i]
        fp = 0
        fn = 0
        for j in range(0,numClasses):
            if (i!=j):
                fp += stats[i][j]
                fn += stats[j][i]
        if (tp+fp) > 0: precision = tp/(tp+fp)
        else: precision = -1
        if (tp+fn) > 0: recall = tp/(tp+fn)
        else: recall = -1
        if precision >= 0 and recall >= 0 and (precision+recall>0): f1 = 2*precision*recall/(precision+recall)        
        else: f1 = -1
        metrics[i]= {'precision': precision,
                     'recall': recall,
                     'f1': f1 }
    return stats,metrics

def printStats(stats):
    cm,m=stats
    print("\nConfusion Matrix")
    print("P \ R  \tnormal\tanomaly")
    print("normal \t"+str(cm[0][0])+"\t"+str(cm[0][1]))
    print("anomaly\t"+str(cm[1][0])+"\t"+str(cm[1][1]))
    print("")
    print("Metrics")
    print("Connection\tPrecision\tRecall\tF-1")
    print("normal  \t"+str(m[0]['precision'])+"\t"+str(m[0]['recall'])+"\t"+str(m[0]["f1"]))
    print("anomaly \t"+str(m[1]['precision'])+"\t"+str(m[1]['recall'])+"\t"+str(m[1]["f1"]))
    print("\n")

def printStatsClassification(stats):
    cm,m=stats
    print("\nConfusion Matrix")
    print("P \ R  \tnormal\tDOS\tU2R\tR2L\tPROBING\tSMURF\tNEPTUNE")
    print("normal \t" + str(cm[0][0]) + "\t" + str(cm[0][1]) + "\t" + str(cm[0][2]) + "\t" + str(cm[0][3]) + "\t" + str(cm[0][4]) + "\t" + str(cm[0][5]) + "\t" + str(cm[0][6]))
    print("DOS    \t" + str(cm[1][0]) + "\t" + str(cm[1][1]) + "\t" + str(cm[1][2]) + "\t" + str(cm[1][3]) + "\t" + str(cm[1][4]) + "\t" + str(cm[1][5]) + "\t" + str(cm[1][6]))
    print("U2R    \t" + str(cm[2][0]) + "\t" + str(cm[2][1]) + "\t" + str(cm[2][2]) + "\t" + str(cm[2][3]) + "\t" + str(cm[2][4]) + "\t" + str(cm[2][5]) + "\t" + str(cm[2][6]))
    print("R2L    \t" + str(cm[3][0]) + "\t" + str(cm[3][1]) + "\t" + str(cm[3][2]) + "\t" + str(cm[3][3]) + "\t" + str(cm[3][4]) + "\t" + str(cm[3][5]) + "\t" + str(cm[3][6]))
    print("PROBING\t" + str(cm[4][0]) + "\t" + str(cm[4][1]) + "\t" + str(cm[4][2]) + "\t" + str(cm[4][3]) + "\t" + str(cm[4][4]) + "\t" + str(cm[4][5]) + "\t" + str(cm[4][6]))
    print("SMURF  \t" + str(cm[5][0]) + "\t" + str(cm[5][1]) + "\t" + str(cm[5][2]) + "\t" + str(cm[5][3]) + "\t" + str(cm[5][4]) + "\t" + str(cm[5][5]) + "\t" + str(cm[5][6]))
    print("NEPTUNE\t" + str(cm[6][0]) + "\t" + str(cm[6][1]) + "\t" + str(cm[6][2]) + "\t" + str(cm[6][3]) + "\t" + str(cm[6][4]) + "\t" + str(cm[6][5]) + "\t" + str(cm[6][6]))