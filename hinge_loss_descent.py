import numpy as np
import matplotlib.pyplot as plt
import pylab

def zero_one_loss(w,x,y,verbose=0):
    if verbose:
        print 'computing zero_one_loss(w =',w,')'
    loss = 0.0
    for (x_,y_) in zip(x,y):
        v = y_*np.dot(w,x_)
        loss = loss + float(v<0)
        if verbose:
            print 'x =',x_,'y =',y_,'loss =',v<0
    loss = loss/np.size(y)
    if verbose:
        print 'total zero_one_loss =',loss
    return loss

def hinge_loss(w,x,y,verbose=0):
    """ evaluates hinge loss and its gradient at w

    rows of x are data points
    y is a vector of labels
    """
    if verbose:
        print 'hinge_loss(w =',w,')'
    loss,grad = 0.0,0.0
    for (x_,y_) in zip(x,y):
        v = y_*np.dot(w,x_)
        loss += max(0,1-v)
        grad += 0 if v > 1 else -y_*x_
        if verbose:
            print 'x =',x_,'y =',y_,'loss =',max(0,1-v)
    loss = loss/np.size(y)
    if verbose:
        print 'total hinge_loss =',loss
        print 'grad = ',grad,'norm(grad) = ',np.linalg.norm(grad)
    return (loss,grad)

def grad_descent(x,y,w,step=0.001,thresh=0.001,grad_thresh=0.0001):
    grad = np.inf
    ws = np.zeros((2,0))
    ws = np.hstack((ws,w.reshape(2,1)))
    step_num = 1
    delta = np.inf
    loss0 = np.inf
    while np.abs(delta)>thresh and step_num < 1/thresh:
        loss, grad = hinge_loss(w,x,y)
        loss0, delta = loss, loss0-loss
        print 'iter %03d: hinge_loss(w)=%0.6f;  norm(grad)=%6f' % (step_num,loss,
            np.linalg.norm(grad)),'1/norm(w)=%0.4f' % (1/np.linalg.norm(w))
        if np.linalg.norm(grad) < grad_thresh:
            return w
        grad_dir = grad/np.linalg.norm(grad)
        w = w-step*grad_dir
        ws = np.hstack((ws,w.reshape((2,1))))
        step_num += 1
    return np.sum(ws,1)/np.size(ws,1)

def test1():
    # sample data points
    x1 = np.array((0  ,.1,.2,.2,.4,.5,.7,.8,.6,0))
    x2 = np.array((0.8,.6,.5,.5,.6,.3,.1,.4,.3,1))
    assert(x1.shape == x2.shape)
    x  = np.vstack((x1,x2)).T
    # sample labels
    y = np.array((1,1,1,1,1,-1,-1,-1,-1,-1))
    assert(x1.shape == y.shape)
    w = grad_descent(x,y,np.array((0,0)),0.1)
    print 'w_opt for hinge_loss is',w
    hinge_loss(w,x,y,1)
    plot_test(w,x,y)

def plot_test(w,x,y):
    plt.figure()
    x1, x2 = x[:,0], x[:,1]
    x1_min, x1_max = np.min(x1)*.7, np.max(x1)*1.3
    x2_min, x2_max = np.min(x2)*.7, np.max(x2)*1.3
    gridpoints = 2000
    x1s = np.linspace(x1_min, x1_max, gridpoints)
    x2s = np.linspace(x2_min, x2_max, gridpoints)
    gridx1, gridx2 = np.meshgrid(x1s,x2s)
    grid_pts = np.c_[gridx1.ravel(), gridx2.ravel()]
    predictions = np.array([np.dot(w,x_) for x_ in grid_pts]).reshape((gridpoints,gridpoints))
    plt.contour(gridx1, gridx2, predictions, levels=[-1, 0, 1])
    plt.contourf(gridx1, gridx2, np.sign(predictions), cmap=plt.cm.Paired)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
    plt.title('1/norm(w)=%0.2f, hinge_loss: %g, zero_one loss: %g' % (1/np.linalg.norm(w),hinge_loss(w,x,y)[0],zero_one_loss(w,x,y)))
    pylab.savefig('../sample_points.png')
    plt.show()

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    test1()
