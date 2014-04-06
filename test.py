f = open('digits.data', 'r')
for x in range(1, 5):
    line = f.readline()
    splits = line.split(',')
    result = splits[64]
    features = splits[:64];
    print features,"count ",len(features),"result ",result