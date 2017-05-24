''' clustering with mixure of guassian '''
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import misc
import glob
from PIL import Image
import colorsys

import os
os.system('cls')

def log_likelihood( data, weight, mean, cov):
    ''' compute likelihood of the given data under founded parameters in\
     gaussian mixure model
     weight : a list of wights of each gaussian
     mean : a list of means( alist of nparray 1D)
     cov : a list of covariances( a list of nparray 2D)
     data : a list of data points or a numpy array

    '''
    # number of clusters
    num_clusters = len(weight)
    # number of dimension
    dim = len(data[0])
    # number of data points
    num_data = len(data)
    log_l = 0
    for i in range(num_data):
        p_d = 0
        for k in range(num_clusters):
            # equivalenly use multivariate_normal.pdf
            n_term1 = np.exp((-.5) * np.dot( (data[i] - mean[k]).T ,\
             np.dot( np.linalg.inv(cov[k]), (data[i] - mean[k]))))
            n_term2 = ((( 2*np.pi)**(num_clusters/2.)) *\
             np.sqrt(np.absolute(np.linalg.det(cov[k]))))**(-1)
            n_term = n_term1 * n_term2
            # likelihood of that data point under a specific gausian
            p_dk = weight[k] * n_term
            # sum over gausiisans = total likelihood of that point
            p_d += p_dk
        # logarithm of likelihood of all data points
        log_l += np.log(p_d)
    return log_l


def e_step(data,weight, mean, cov):
    ''' E step of EM: compute responsibility for a given parameters'''
    num_data = len(data)
    num_clusters = len(weight)
    resp = np.zeros((num_data, num_clusters))
    for i in range(num_data):
        for k in range(num_clusters):
            # responsibility of k cluster for ith data point
            resp[i,k] = weight[k] * multivariate_normal.pdf(data[i] , mean[k], cov[k])
    # normalize resposibility so that sum of res of all clusters for a dtpt os 1
    row_sum =  np.sum(resp, axis=1)
    # check for float results
    resp = resp / row_sum[:,None]
    return resp


def compute_weight(resp):
    ''' M step: compute weights of each cluster'''
    # soft counts = effective number of observations in each cluster
    n_k = np.sum(resp, axis=0)
    num_data = resp.shape[0]
    weight = n_k / float(num_data)
    return weight

def compute_mean(resp, data):
    ''' M step: for a given resposibility, compute means of all gaussian'''
    n_k = np.sum(resp, axis=0)
    mean = np.dot(resp.T , data) / n_k[:, np.newaxis]
    return mean


def compute_cov(resp, data, mean):
    ''' M step: for a given resposibility, compute covariances of all gaussian'''
    num_data = len(data)
    num_dim = len(data[0])
    num_cluster = resp.shape[1]
    n_k = np.sum(resp, axis=0)
    cov = []
    for k in range(num_cluster):
        a = np.zeros((num_dim,num_dim))
        for i in range(num_data):
            a += resp[i,k] * np.outer( (data[i] - mean[k]) , (data[i] - mean[k]))
        cov_k = a / float(n_k[k])
        cov.append(cov_k)
    return(cov)


def EM(data, initial_mean ,initial_weight, initial_cov, max_iter = 1000, tolerance = 1e-4):
    ''' EM algorithm for solving gaussian mixure model
    OUTPUT: is a dictionary of :
    gaussian parameters, responsibility,
    and loglikelihood as the algorithm convereges
    '''
    # make a copy of initial parameters
    weight = initial_weight[:]
    mean = initial_mean[:]
    cov = initial_cov[:]

    # infer number of data points and clusters
    num_data = len(data)
    num_clusters = len(initial_weight)

    # initialize some usefull variables
    resp = np.zeros((num_data, num_clusters))
    ll_prev = log_likelihood( data, weight, mean, cov)
    ll_history = [ll_prev]

    for iter in range(max_iter):

        # E step
        resp = e_step(data,weight, mean, cov)
        # M step
        weight = compute_weight(resp)
        mean = compute_mean(resp, data)
        cov =  compute_cov(resp, data, mean)
        # compute log likelihood at this iteration
        ll = log_likelihood( data, weight, mean, cov)
        ll_history.append(ll)

        # check stopping condition
        if ll - ll_prev < tolerance:
            print iter
            break
        # if the stopping condition does not satisfy do this and continue
        ll_prev = ll
        # number of iteration that convergence reach
    print " convergence reach after iterations" , iter

    return {'weights': weight,\
            'means' :  mean,\
            'covariances' : cov,
            'log_likelihood' : ll_history,
            'responsibility' : resp}

## The first data set: this part is to check our code
## first generate som data from gaussians and then use Em to cluster these data
# def generate_MoG_data(num_data, means, covariances, weights):
#     num_clusters = len(weights)
#     data = []
#     for i in range(num_data):
#         k = np.random.choice(num_clusters,1, p = weights)[0]
#         a = np.random.multivariate_normal(means[k], covariances[k])
#         data.append(a)
#     return data
#
# init_means = [
#     [5, 0], # mean of cluster 1
#     [1, 1], # mean of cluster 2
#     [0, 5]  # mean of cluster 3
# ]
# init_covariances = [
#     [[.5, 0.], [0, .5]], # covariance of cluster 1
#     [[.92, .38], [.38, .91]], # covariance of cluster 2
#     [[.5, 0.], [0, .5]]  # covariance of cluster 3
# ]
# init_weights = [1/4., 1/2., 1/4.]  # weights of each cluster
# # Generate data
# np.random.seed(4)
# data = generate_MoG_data(100, init_means, init_covariances, init_weights)
#
# for i in range(100):
#     plt.scatter(data[i][0], data[i][1])
#
#
# # feed the above data to EM
# ## initial weight, means and covariances
# np.random.seed(4)
# initial_weight = [1./3]*3
# initial_mean_indices = np.random.choice(len(data), 3, replace=False)
# initial_mean = [ data[x] for x in initial_mean_indices]
# initial_cov = [np.cov(data, rowvar = False)]*3
#
# results = EM(data, initial_mean ,initial_weight, initial_cov, max_iter = 1000)
# print "weights:"
# print results['weights']
# print " means : "
# print results['means']
# print " covariances :"
# print results['covariances']
# plt.figure()
# plt.plot(results['log_likelihood'], linewidth = 4)
# plt.xlabel('Iteration')
# plt.ylabel('log likelihood')
#
#
# # plot contour plots and observe the progress of parameters through \
# # plotting datapoints and the contour plots of the founded clusters
# def plot_contours(data, means, covs, title):
#     plt.figure()
#     plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data
#
#     delta = 0.025
#     k = len(means)
#     x = np.arange(-2.0, 7.0, delta)
#     y = np.arange(-2.0, 7.0, delta)
#     X, Y = np.meshgrid(x, y)
#     col = ['green', 'red', 'indigo']
#     for i in range(k):
#         mean = means[i]
#         cov = covs[i]
#         sigmax = np.sqrt(cov[0][0])
#         sigmay = np.sqrt(cov[1][1])
#         sigmaxy = cov[0][1]/(sigmax*sigmay)
#         Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
#         plt.contour(X, Y, Z, colors = col[i])
#         plt.title(title)
#     plt.rcParams.update({'font.size':16})
#     plt.tight_layout()
#
# # contour plot for the initializaion parameters
# plot_contours(data, initial_mean, initial_cov, 'initial clusters')
#
# # contour plot after 12 iteration
# results_12 = EM(data, initial_mean ,initial_weight, initial_cov, max_iter = 12)
# plot_contours(data, results_12['means'], results_12['covariances'], 'clusters after 12 iteration')
#
# # contour plot after convergence of algorithm which 23 iteration
# plot_contours(data, results['means'], results['covariances'], 'final clusters')
#
# # contour plot for the originaal gaussians which we took data from
# plot_contours(data, init_means, init_covariances, 'the real distribution')
# plt.show()

## socound data points : image data
# load images
filename_list = []
all_images = []
for filename in glob.glob('E:\learn_python\clustring\week4\images/*.jpg'):
    filename_list.append(filename)
    #im = misc.imread(filename)
    im = Image.open(filename)
    rgb_im = im.convert('RGB')
    total_rgb = np.mean(rgb_im, axis=(0,1))
    rgb = total_rgb / 256.
    all_images.append(rgb)
print all_images[0]
print filename_list[0]

# initialization
# 4 is number of clusters and 3 is the dimnetion(rgb)
np.random.seed(1)
init_weight = [1./4, 1./4, 1./4, 1./4]
initial_mean_indices = np.random.choice(len(all_images), 4, replace = False)
initial_mean = [all_images[x] for x in initial_mean_indices]
initial_diag_cov = np.zeros((3,3))
for i in range(3):
    initial_diag_cov[i,i] = np.var([x[i] for x in all_images])
initial_cov = [initial_diag_cov]*4

#run the EM algorithm on the image data using the above initialization
out = EM(all_images, initial_mean ,init_weight, initial_cov)
print "weights:"
print out['weights']
print " means : "
print out['means']
print " covariances :"
print out['covariances']
plt.figure()
plt.plot(out['log_likelihood'], linewidth = 4)
plt.xlabel('Iteration')
plt.ylabel('log likelihood for images')
#
print out['responsibility'][0]

# evaluating changing responsibility and uncertainty as iterations continue.
def plot_responsibilities_in_RB(img, resp, title):
    N, K = resp.shape

    HSV_tuples = [(x*1.0/K, 0.5, 0.9) for x in range(K)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    R = [x[0] for x in img]
    B = [x[1] for x in img]
    resp_by_img_int = [[resp[n][k] for k in range(K)] for n in range(N)]
    cols = [tuple(np.dot(resp_by_img_int[n], np.array(RGB_tuples))) for n in range(N)]

    plt.figure()
    for n in range(len(R)):
        plt.plot(R[n], B[n], 'o', c=cols[n])
    plt.title(title)
    plt.xlabel('R value')
    plt.ylabel('B value')
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()

plot_res = 0
if plot_res == 1:
    # plot random responsibility
    N, K = out['responsibility'].shape
    random_resp = np.random.dirichlet(np.ones(K), N)
    plot_responsibilities_in_RB(all_images, random_resp, 'Random responsibilities')
    # plot responsibility after 1 itertation
    out_1 = EM(all_images, initial_mean ,init_weight, initial_cov, max_iter = 1)
    plot_responsibilities_in_RB(all_images, out_1['responsibility'], 'After 1 iteration')
    # plot responsibility after 20 iterations
    out_20 = EM(all_images, initial_mean ,init_weight, initial_cov, max_iter = 20)
    plot_responsibilities_in_RB(all_images, out_20['responsibility'], 'After 20 iteration')
    plt.show()


def show_top_image(clus, m=5):
    ''' show the top images in each cluster'''
    # get the indices that has the higher likelihood in each clusters
    indices_top_clus = np.argsort(out['responsibility'][:,clus])[-m:][: : -1]
    for ind in indices_top_clus:
        print all_images[ind]
        img = Image.open(filename_list[ind])
        img.show()

for clus in range(4):
    show_top_image(clus, m=5)
