import numpy as np  
  
def nonlin(x,deriv=False):  
    if(deriv==True):  
        return x*(1-x)  
  
    return 1/(1+np.exp(-x))  
      
X = np.array([[0,0,1],  
            [0,1,1],  
            [1,0,1],  
            [1,1,1]]) 
# print X.shape
                  
y = np.array([[0],  
            [1],  
            [1],  
            [0]])  
# print y.shape
np.random.seed(1)  
  
# randomly initialize our weights with mean 0  
w0 = 2*np.random.random((3,4)) - 1  
w1 = 2*np.random.random((4,1)) - 1
# print w0
# print w1
# print w0.shape
# print w1.shape
  
for j in range(60000):
  
     
    l0 = X  
    l1 = nonlin(np.dot(l0,w0))  
    l2 = nonlin(np.dot(l1,w1))  
  
      
    l2_error = y - l2  
      
    if (j% 10000) == 0:  
        print("Error:" + str(np.mean(np.abs(l2_error))))
          
     
    l2_delta = l2_error*nonlin(l2,deriv=True)  
  
     
    l1_error = l2_delta.dot(w1.T)  
      
      
    l1_delta = l1_error * nonlin(l1,deriv=True)  
  
    w1 += l1.T.dot(l2_delta)  
    w0 += l0.T.dot(l1_delta)  