import numpy as np

def gen_a_mat(n):
    A = np.random.rand(n**2).reshape(n, n)
    return A

def generate_mat ( n, T, block, filename='', is_test = False ):
    A = np.zeros((T,block,n,n))
    for i in range(T):
        for j  in range(block):
            A[i,j] = gen_a_mat(n)
    
    if is_test:
        return A
    else:
        np.save(filename,A)



def generate_points( mfd , n, T, block, filename='', is_test = False):
    A = np.zeros((T,block,n,n))
    for i in range(T):
        for j  in range(block):
            A[i,j] = mfd.rand()
            #A[i,j] = point / np.linalg.norm(point)
    if is_test:
        return A
    else:
        np.save(filename,A)
        

def normalize(mfd,point):
    vec = mfd.log(mfd.center,point)
    nor_vec = vec / mfd.norm(mfd.center,vec)
    return mfd.exp( mfd.center, nor_vec )
