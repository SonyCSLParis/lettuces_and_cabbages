from functools import reduce
from operator import xor
from itertools import chain

set1 = [[0.1, 0.2], [0.4, 0.6], [0.65, 0.66], [0.8, 1]]
set2 = [[0, 1]]

l = sorted((reduce(xor, map(set, chain(set1 , set2))))) 

res = [l[i:i + 2] for i in range(0, len(l), 2)]

def merge_intervals(a):
  b = []
  for ai in a:
     if b and b[-1][1] > ai[0]:
         b[-1][1] = max(b[-1][1], ai[1])
     else:
         b.append([ai[0], ai[1]])
  return b


#a=[[59.814207273518754, 60.64653155368713], [59.663615818664596, 61.848065414009625], [60.85354054482041, 62.529752855907454], [62.53969783241004, 63.288591656577495], [62.269323046107544, 64.46715677946413], [63.34044029038293, 64.24265144890387]]
a= [[ -0.8, 0.8],
    [11.3956767, 14.59513818],
    [99.2, 100.8       ]]

b = merge_intervals(a)

print(union)