from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import scipy
import pylab

def calc_square_error(w,x,y):
    '''Calculates the mean square error from w*x to y'''
    return mean(power(dot(x,w)-y,2)) 
    
def solve_ls(x,y):
    '''Find w for the formula dot(x, w)=y'''
    w = dot(scipy.linalg.pinv(x),y)
    return w

def polynomial(x1, x2, degree=3, gamma=1 , coeff0=0):
    ''' Returns the evaluation of polynomial kernel between all pairs of vectors in x1 and x2.
    x1,x2 - arrays of size n1xd and n2xd where n1 and n2 are the number of vectors in x1 and x2 
    and d is the vectors' dimension 
    gamma - the polynomial coefficient(1 by default)
    degree - the polynomial degree (3 by default)
    coeff0 - free parameter (0 by default)
    '''
    return matrix(power(gamma*(x1.dot(x2.T)) + coeff0, degree))
    
def create_cross_validation_idxs(num_samples, num_folds):
    '''Creates num_folds different and foreign folds of the data.
    This method returns a collection of (training_samples_idxs, validation_samples_idxs) pairs,
    every pair must have a single, different fold as the validation set, and the other folds as training.
    PICK THE ELEMENTS OF EACH FOLD RANDOMLY. The collection needs to have num_folds such pairs.'''

    # generate a random permutation of indices
    perm = random.permutation(num_samples)
    # size of each fold
    fold_size = ceil(num_samples/num_folds)
    assert(fold_size>1)
    result = []
    validation_samples_idxs = []
    training_samples_idxs = []
    for i in range(num_folds):
        # validation fold indices
        val_fold_idxs = arange(i*fold_size,(i+1)*fold_size,dtype=int)
        # training folds indices
        train_folds_idxs = r_[arange(i*fold_size,dtype='int'), arange((i+1)*fold_size,num_samples,dtype=int)]
        # take
        validation_samples_idxs.append(take(perm,val_fold_idxs))
        training_samples_idxs.append(take(perm,train_folds_idxs))
    return zip(training_samples_idxs, validation_samples_idxs)

def load_data():
    dataset = load_boston()
    num_samples = size(dataset.data, 0)
    test_set_sz = int(1.0*num_samples/10)
    tst_sub_inds = random.choice(range(num_samples), test_set_sz, False)
    data_test, label_test = dataset.data[tst_sub_inds, :], dataset.target[tst_sub_inds]
    trn_sub_inds = list(set(range(num_samples))-set(tst_sub_inds)) 
    data_train, label_train = dataset.data[trn_sub_inds, :], dataset.target[trn_sub_inds]
    return ((data_train, label_train), (data_test, label_test))
    
if __name__=='__main__':

    ((data_train, label_train), (data_test, label_test)) = load_data()
    
    num_of_lambdas, k_fold, fold = 100, 8, 0
    lambda_vals = linspace(1e2, 1e20, num_of_lambdas)
    errs = zeros([num_of_lambdas, k_fold, 2])
    for (train, vld) in create_cross_validation_idxs(size(label_train), k_fold):
        for lambda_ind,lambda_val in enumerate(lambda_vals):
            #sub-divide to training and validation sets:
            data_vld, label_vld = data_train[vld, :], label_train[vld]
            data_train_fold, label_train_fold = data_train[train, :], label_train[train]

            gram_mat = polynomial(data_train_fold, data_train_fold, 3)
            gram_vld = polynomial(data_vld, data_train_fold, 3)

            alpha_opt = linalg.pinv(gram_mat+.5*lambda_val*eye(gram_mat.shape[0])).dot(label_train_fold).T

#            print linalg.norm(alpha_opt)
            y_regress = gram_vld.dot(alpha_opt)
            y_train   = gram_mat.dot(alpha_opt)

            errs[lambda_ind, fold, 0] = mean(power(y_regress-label_vld,2))
            errs[lambda_ind, fold, 1] = mean(power(y_train-label_train_fold,2))
        fold = fold+1
    best_lambda_idx = argmin(mean(errs[:,:,0], axis=1))
    best_lambda = lambda_vals[best_lambda_idx]
    best_lambda_error = min(mean(errs[:,:,0], axis=1))
    
    #Plot training and validation errors as a function of lambda, as well as the final test error for best_lambda
    fig = plt.figure()
    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except:
        pass

    ax = fig.gca()
    ax.set_xscale('log')
    plt.plot(lambda_vals, mean(errs[:,:,0],axis=1),label=r'Mean validation error')
    plt.plot(lambda_vals, mean(errs[:,:,1],axis=1),label=r'Mean training error')

    plt.plot(best_lambda, best_lambda_error, 'r^')
    plt.xlabel(r'values of \lambda on log scale')
    plt.ylabel(r'mean squared prediction error')
    plt.title(r"best_lambda=%0.2f" % best_lambda, fontsize=16, color='gray')
    ax.legend()
    pylab.savefig('ridge_regression.png')
    plt.show()
