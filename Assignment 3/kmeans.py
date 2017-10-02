'''
Created on Oct 1, 2017

@author: aqd14
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def init_centroids(A, n_clusters):
    """
    Initialize centroids for pixels in RGB mode (ranging from 0 to 255).
    Randomly pick n_clusters points in the original image to be centroids.
    
    Parameters
    ----------
    n_clusters : int
        Number of expected clusters
    A : 3-d matrix
        The pixels in image and their coordinates
    Returns
    -------
    centroids : array-like
        A randomly initialized centroids ranging from 0 to 255
    """
    centroids = A[np.random.choice(A.shape[0], n_clusters, replace=False),
                  np.random.choice(A.shape[1], n_clusters, replace=False), :]
    
    return centroids

def init_cluster(centroids):
    """Initialize cluters

    Parameters
    ----------
    centroids : 2-d array
        List of centroids

    Returns
    -------
    clusters : dictionary
        Mapping from centroids to a list of points in clusters
    """
    clusters = {}
    
    for c in range(centroids.shape[0]):
        clusters[c] = []
    return clusters
    
def assign_cluster(A, clusters, centroids):
    """Assign nearest cluster for all pixels
    
    Parameters
    ----------
    A : RBG matrix representation for image

    clusters : dictionary
        List of centroids associated with their points in clusters

    centroids : 2-d array
        Current centroids
    """
    # Euclid distance from given point to the centroids
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
#             assign_cluster(clusters, centroids, A[i][j])
            pixel = A[i][j]
            distance = np.sum((centroids - pixel) ** 2, axis=1)
            # Assign closest cluster for the given pixel
            min_index = np.argmin(distance)
            clusters[min_index].append(pixel)
            
def update_centroids(clusters, centroids):
    new_centroids = np.zeros((centroids.shape[0], centroids.shape[1]))
    for c in range(centroids.shape[0]):
        points = np.asarray(clusters[c])
        if len(points) > 0:
            new_centroids[c] = np.round(np.mean(points, axis=0))
        else:
            # a centroid without any points
            new_centroids[c] = centroids[c]
    return new_centroids

def kmeans(A, n_clusters, max_iter=100, tolerance=1e-5):
    """Simple implementation for K-Means algorithm to compress an image by reducing the number of colors it contains
    
    Parameters
    ----------
    A : RGB matrix representation for image
    
    n_clusters : int
        Number of color clusters
        
    max_iter : int
        Maximum number of iteration for finding centroids
        Default value is 100
        
    tolerance : float
        The minimum Euclid distance of centroids values between two consecutive iteration to be considered converged
         
    Returns
    -------
    centroids : 2-d array
        Converged centroids
    """
    centroids = init_centroids(A, n_clusters)  # default centroids
    clusters = init_cluster(centroids)  # np.zeros((A.shape[0], A.shape[1], 1))    # store the index of centroids for each pixel
    ite = 1
    while(ite <= max_iter):
        # print('Iteration {0}'.format(ite))
        assign_cluster(A, clusters, centroids)
        update_centroids(clusters, centroids)
        new_centroids = update_centroids(clusters, centroids)
        
        err = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        # print('Error = {0}\n'.format(err))
        if err < tolerance:
            print('Converged after {0} iterations!'.format(ite))
            break;
        centroids = new_centroids
        ite += 1
    
    return centroids

def compress_image(B, centroids):
    """Replace each pixel in the image with its nearest cluster centroid color
    
    Parameters
    ----------
    centroids : 2-d array
        Convergered centroids
    
    B : RBG matrix image
        Image to be compressed
    
    Returns
    -------
    B : RBG matrix image
        Compressed image
    """
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            pixel = B[i][j]
            distance = np.sum((centroids - pixel) ** 2, axis=1)
            # Assign closest cluster for the given pixel
            min_index = np.argmin(distance)
            B[i][j] = centroids[min_index]
    
def main():
    A = misc.imread('pa3data/b_small.tiff', mode='RGB')
    centroids = kmeans(A, n_clusters=16)
    # print(centroids)
    B = misc.imread('pa3data/b.tif', mode='RGB')
    compress_image(B, centroids)
    
    plt.imshow(B)
    # plt.show()
    plt.savefig('figures/kmeans.png')
    
    for n_clusters in range(2, 16):
        centroids = kmeans(A, n_clusters)
        print('Centroid for clusters {0} are {1}'.format(n_clusters, centroids))
        compress_image(B, centroids)
        plt.imshow(B)
        plt.savefig('figures/kmeans' + str(n_clusters) + '.png')
      
if __name__ == '__main__':
    main()
