import numpy as np
import math
A = np.array([
  [-2,1,0],
  [2,-2,0],
  [1,-1,-5]])

#A = np.array([
#  [-2,1],
#  [2,-2]])

PS = np.array([[2,1,-3],[1,0,-3],[-3,-3,3]])

#PS = np.array([[2,1],[1,0]])

T = 2
max = 3
def gen(T, max):
  T = T + 1
  ys = np.zeros((int(math.pow(max,T)),T), dtype='int32')
  for t in range(T):
    step = 0
    for row in range(int(math.pow(max,T))):
      l = step % max
      ys[row,t] = l
      if row % math.pow(max,t) == math.pow(max,t)-1:
        step += 1
  return ys

x = [0, 1]
#ys = [[2,0, 1,2],[2,1, 1,2],[2,0, 0,2],[2,1, 0,2]]
#ys = [[0, 1],[1, 1],[0, 0],[1, 0]]
ys = gen(T, max)
y = np.array([2, 0, 1])

def s(T,y):
  sum = 0
  for t in range(1,T+1):
    sum += A[y[t-1],y[t]] + PS[t-1,y[t]]
  return sum

def g(a,b,i):
  return math.exp(A[a,b] + PS[i,b])

def get_M(i, max):
  M = np.zeros((max,max))
  for j in range(max):
    for k in range(max):
      M[j,k] = g(j,k,i)
  return M

def Z(max, T):
  M = np.identity(max)
  for i in range(T):
    M = np.matmul(M, get_M(i, max))
  return M.sum()

z = 0
for i in range(len(ys)):
  z += math.exp(s(T, ys[i,:]))

print(z)

sum = 0
for i in range(len(ys)):
  p = math.exp(s(T, ys[i,:])) / z
  #print(p)
  sum += p

print(sum)

sc = 1
for t in range(T):
  sc *= g(y[t],y[t+1], t)

print(sc)


