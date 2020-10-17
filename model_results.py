
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
    print("P \ R\tnormal\tanomaly")
    print("normal\t"+str(cm[0][0])+"\t"+str(cm[0][1]))
    print("anomaly\t"+str(cm[1][0])+"\t"+str(cm[1][1]))
    print("")
    print("Metrics")
    print("Conexion\tPrecision\tRecall\tF-1")
    print("normal\t"+str(m[0]['precision'])+"\t"+str(m[0]['recall'])+"\t"+str(m[0]["f1"]))
    print("anomaly\t"+str(m[1]['precision'])+"\t"+str(m[1]['recall'])+"\t"+str(m[1]["f1"]))
    print("\n")
  