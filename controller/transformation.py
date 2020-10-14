# Library to transform data
def transformData(kddCup):
    # Transform Text data to number
    protocolTypeLabel = {'icmp': 0, 'tcp': 1, 'udp': 2}
    kddCup["protocol_type"] = transformColumn(kddCup["protocol_type"],protocolTypeLabel,0)
    # print(kddCup["protocol_type"])
    flagLabel = {'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4, 'S0': 5, 'S1': 6, 'S2': 7, 'S3': 8, 'SF': 9, 'SH': 10}
    kddCup["flag"] = transformColumn(kddCup["flag"],flagLabel,0)
    # print(kddCup["flag"])
    transformLabel = {'normal': 0}
    kddCup["label"] = transformColumn(kddCup["label"],transformLabel,1)
    # print(kddCup["label"])
    return kddCup


def transformColumn(column, array, default):
   rows = list(set(column.tolist()))
   for row in rows:
      try:
         column = column.replace([row],array[row])
      except KeyError:
         column = column.replace([row],default)
   return column
