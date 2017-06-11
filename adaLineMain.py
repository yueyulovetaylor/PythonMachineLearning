import sys
sys.path.append('Utilities/')
import readData as RD

from adaLine import adaLineGD

print('Implementation of AdaLine Learning')

DataMap = RD.readDataFromIris()
X = DataMap['X']
y = DataMap['y']

# Compare to AdaLine learning by changing eta between 10e-2 and 10e-4
print('construct two adaLine model with eta 0.01 and 0.0001')
ada1 = adaLineGD(eta = 0.01, n_iter = 10)
ada2 = adaLineGD(eta = 0.0001, n_iter = 10)