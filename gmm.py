''' clustering with mixure of guassian '''
import numpy as np
from scipy.stats import multivariate_normal
import sklearn.metrics.pairwise
import os
os.system('cls')

def log_likelihood( data, weight, mean, cov):
    ''' compute likelihood of the given data under founded parameters in\
     gaussian mixure model
    '''
    # number of clusters
    num_clusters = len(weight)
    # number of dimension
    dim = data.shape[1]
    log_l = 0
    for i in range(data.shape[0]):
        p_d = 0
        for k in range(num_clusters):

            n_term1 = np.exp((-.5) * np.dot( (data[i] - mean[k]).T ,\
             np.dot( np.linalg.inv(cov[k]), (data[i] - mean[k]))))
            n_term2 = ((( 2*np.pi)**(num_clusters/2.)) *\
             np.sqrt(np.absolute(np.linalg.det(cov[k]))))**(-1)
            n_term = n_term1 * n_term_2
            p_dk = weight[k] * n_term
            p_d += p_dk

        log_l += np.log(p_d)
    return log_l

def e_step2(data, weight, mean, cov):
    num_data = data.shape[0]
    num_clusters = len(weight)
    resp = np.zeros((num_data, num_clusters))
    for i in range(num_data):
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            n_term1 = np.exp((-.5) * np.dot( (data[i] - mean[k]).T ,\
             np.dot( np.linalg.inv(cov[k]), (data[i] - mean[k]))))
            n_term2 = ((( 2*np.pi)**(num_clusters/2.)) *\
             np.sqrt(np.absolute(np.linalg.det(cov[k]))))**(-1)
            n_term = n_term1 * n_term2
            p_dk = weight[k] * n_term
            Z[k] = p_dk
        resp[i,:] = Z / float(sum(Z))
    return resp


def e_step(data,weight, mean, cov):
    num_data = data.shape[0]
    num_clusters = len(weight)
    resp = np.zeros((num_data, num_clusters))
    for i in range(num_data):
        for k in range(num_clusters):
            resp[i,k] = weight[k] * multivariate_normal.pdf(data[i] , mean[k], cov[k])
    row_sum =  np.sum(resp, axis=1)
    # check for float results
    resp = resp / row_sum[:,None]
    return resp


def compute_weight(resp):
    n_k = np.sum(resp, axis=0)
    num_data = resp.shape[0]
    weight = n_k / float(num_data)
    return weight

def compute_mean(resp, data):
    n_k = np.sum(resp, axis=0)
    mean = np.dot(resp.T , data) / n_k[:, np.newaxis]
    return mean


def compute_cov(resp, data, mean):
    num_data = data.shape[0]
    num_dim = data.shape[1]
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


    # if initial_mean is None:
    #     rand_indices = np.random.randint(0, num_data, num_clusters)
    #     initial_mean = data[rand_indices]
    #
    #         resp = np.zeros((num_data, num_clusters))
    #         dis = sklearn.metrics.pairwise.euclidean_distances(data, initial_mean)
    #         membership = np.argmin(dis, axis=1)
    #         resp[ [np.arange(num_data)] , membership ] = 1.

def EM(data, initial_mean ,initial_weight, initial_cov, max_iter = 1000, tolerance = 1e-4):

    weight = initial_weight[:]
    mean = initial_mean[:]
    cov = initial_cov[:]

    num_data = data.shape[0]
    num_clusters = len(initial_weight)

    resp = np.zeros((num_data, num_clusters))

    ll_prev = log_likelihood( data, weight, mean, cov)
    ll_history = [ll_prev]

    for iter in range(max_iter):
        print " iter" , iter
        # e step
        resp = e_step(data,weight, mean, cov)
        # mstep
        weight = compute_weight(resp)
        mean = compute_mean(resp, data)
        cov =  compute_cov(resp, data, mean)

        ll = log_likelihood( data, weight, mean, cov)
        ll_history.append(ll)

        if ll - ll_prev < tolerance:
            print iter
            break
        ll_prev = ll

    return {'weights': weight,\
            ' means' :  mean,\
            'covaariances' : cov,
            'log_likelihood' : ll_history'
            'responsibility' : resp}
