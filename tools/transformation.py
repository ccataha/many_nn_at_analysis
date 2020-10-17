# Library to transform data
def transformData(kddCup):
    # Transform Text data to number
    kddCup["protocol_type"] = transformColumn(kddCup["protocol_type"],protocolTypeLabel(),2)
    # print(kddCup["protocol_type"])
    kddCup["flag"] = transformColumn(kddCup["flag"],flagsLabel(),1)
    # print(kddCup["flag"])
    kddCup["service"] = transformColumn(kddCup["service"],serviceLabel(),19)          
    transformLabel = {'normal': 0}
    kddCup["label"] = transformColumn(kddCup["label"],transformLabel,1)
    # print(kddCup["label"])
    return kddCup

# Library to transform data
def transformDataClassification(kddCup):
    # Transform Text data to number
    kddCup["protocol_type"] = transformColumn(kddCup["protocol_type"],protocolTypeLabel(),2)
    # print(kddCup["protocol_type"])
    kddCup["flag"] = transformColumn(kddCup["flag"],flagsLabel(),1)
    # print(kddCup["flag"])
    kddCup["service"] = transformColumn(kddCup["service"],serviceLabel(),19) 
    kddCup["label"] = transformColumn(kddCup["label"],transfromLabel(),3)
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



def protocolTypeLabel():
    return {'icmp': 0, 'tcp': 1, 'udp': 2}

def flagsLabel():
    return {'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4, 'S0': 5, 'S1': 6, 'S2': 7, 'S3': 8, 'SF': 9, 'SH': 10}


def serviceLabel():
    return {'auth': 0, 'bgp': 1, 'courier': 2, 'csnet_ns': 3, 'ctf': 4,'daytime': 5, 'discard': 6,
               'domain': 7, 'domain_u': 8, 'echo': 9, 'eco_i': 10, 'ecr_i': 11, 'efs': 12,
               'exec': 13, 'finger': 14, 'ftp': 15, 'ftp_data': 16, 'gopher': 17, 'hostnames': 18,
               'http': 19, 'http_443': 20, 'imap4': 21, 'IRC': 22, 'iso_tsap': 23, 'klogin': 24, 'kshell': 25,
               'ldap': 26, 'link': 27, 'login': 28, 'mtp': 29, 'name': 30, 'netbios_dgm': 31, 'netbios_ns': 32,
               'netbios_ssn': 33, 'netstat': 34, 'nnsp': 35, 'nntp': 36, 'ntp_u': 37, 'other': 38,
               'pm_dump': 39, 'pop_2': 40, 'pop_3': 41, 'printer': 42, 'private': 43, 'red_i': 44,
               'remote_job': 45, 'rje': 46, 'shell': 47, 'smtp': 48, 'sql_net': 49, 'ssh': 50,
               'sunrpc': 51, 'supdup': 52, 'systat': 53, 'telnet': 54, 'tftp_u': 55, 'tim_i': 56, 'time': 57,
               'urh_i': 58, 'urp_i': 59, 'uucp': 60, 'uucp_path': 61, 'vmnet': 62, 'whois': 63,
               'X11': 64, 'Z39_50': 65}

def transfromLabel():
    # DOS-1   866
    # U2R-2   519
    # R2L-3    140
    # PROBING-4  519  
    # SMURF-5   280790
    # NEPTUNE-6  107201
    return {'normal': 0, 'back': 1, 'buffer_overflow': 2, 'ftp_write': 3,
               'guess_passwd': 3, 'imap': 3, 'ipsweep': 4, 'land': 1, 'loadmodule': 2, 'multihop': 3,
               'neptune': 6, 'nmap': 4, 'perl': 2, 'phf': 3, 'pod': 1, 'portsweep': 4,
               'rootkit': 2, 'satan': 4, 'smurf': 5, 'spy': 3, 'teardrop': 1, 'warezclient': 3, 'warezmaster': 3}