# Main program to run colour clustering
from PIL import Image
import numpy as np
import sys


""" Find k-representative colours of the image
"""

def get_sq_distance(img, means, pseudo_flag=False):
    """ Returns the squared distances between each pixel in the image 'img'
    and each of the means in 'means'. If pseudo_flag=True, the distance terms
    that are only dependent on 'img' (i.e. a_sq, see below) are ignored.

    Squared distance calculation:
        sq_dist = sum_i ( a_i - b_i )^2, where i is a dimension
                = inner_product( vec(a)-vec(b), vec(a)-vec(b) )
                = vec(a)^2 - 2*inner_prod( vec(a), vec(b) ) + vec(b)^2
                -> a_sq - 2*ab + b_sq

    Pseudo-squared distance calculation:
    Exactly the same as the square distance calculation, except that a_sq is
    treated as a constant=0. This is good for doing relative distance
    comparisons.
    """
    # Calculate b_sq
    # Dependent on if we have one mean or an array of means.
    if means.ndim == 1:
        ab = np.expand_dims(np.matmul(img, means.T), axis=1)
        b_sq = np.array([[np.sum(np.power(means,2))]])
    else:
        ab = np.matmul(img, means.T)
        b_sq = np.sum(np.power(means, 2), axis=1)
        b_sq = np.expand_dims(b_sq, axis=0)

    # Calculate distance
    if pseudo_flag:
        # Calculate pseudo squared distance
        dist = -2*ab + b_sq
    else:
        # Calculate squared distance
        a_sq = np.sum(np.power(img.astype(np.float32), 2), axis=1)
        a_sq = np.expand_dims(a_sq, axis=1)
        dist = a_sq - 2*ab + b_sq

    return dist


#TODO: Could introduce random restarts
#TODO: Compare convergence. Suspect overall distance-to-means of kmeans is high
def kmeans_driver(img, means):
    """ Run the Kmeans algorithm on the image 'img' with the initial means,
    'means'.

    Returns:
        cluster_num: Vector of integers. Suppose cluster_num[i] = j, then pixel
            i of 'img' is closest to the cluster mean, means[j].
        means: 2D array. Each row j is the mean of a cluster of pixels.

    """
    # Calculations for means must be in floats
    if means.dtype != 'float':
        means = means.astype(np.float32)

    # For each cluster, compare Euclidean distance from pixel
    k = len(means)
    old_cluster_num = np.random.randint(k, size=len(img))
    similarity = 0
    loop_counter = 0

    while similarity < 1.0:
        # Calculate the relative distances between each pixel and each mean
        pseudo_dist = get_sq_distance(img, means, True)
        
        # Find index of the smallest distance
        cluster_num = np.argmin(pseudo_dist, axis=1)

        # Update cluster means
        #TODO: worry about potential divide by zero in np.mean
        means = np.zeros((k, 3), dtype=np.float32)
        for ii in xrange(k):
            cluster_members = img[cluster_num==ii, :]
            means[ii,:] = np.mean(cluster_members, axis=0)

        # Monitor convergence; see if the clustering has changed
        similarity = np.sum(old_cluster_num == cluster_num) / float(len(cluster_num))
        old_cluster_num = cluster_num
        loop_counter += 1
  
    print("Number of loops needed until convergence: {0}".format(loop_counter))
    means = (np.round(means)).astype(np.uint8)
    return cluster_num, means

def kmeans(img, k):
    """ Colour clusters image 'img' to 'k' number of colours. This is done with
    the Kmeans algorithm.
    
    Selects k-initial means uniformly randomly, without replacement. Pass
    these initial means to the kmeans_driver(..).
    """
    # Randomly pick k pixels as initial cluster "means"
    # Random indices are picked without replacement; to avoid duplicate means
    n = len(img) 
    rand_ind = np.random.choice(n, size=k, replace=False) 
    means = img[rand_ind, :].astype(np.float32) 

    print "Using Kmeans.."
    return kmeans_driver(img, means)

def kmeans_pp(img, k):
    """ Colour clusters image 'img' to 'k' number of colours. This is done with
    the Kmeans++ algorithm.
    
    Determines initial pixel means using Kmeans++. Pass these initial means to
    the kmeans_driver(..).
    """
    n = len(img)
    means = np.zeros((k, 3), dtype=np.float32)
    rand_ind = np.random.randint(n)
    means[0,:]= img[rand_ind,:].astype(np.float32)

    # Pick means based on a probability distribution
    dist_mat = np.inf * np.ones((n, k))
    pseudo_dist = get_sq_distance(img, means[0,:])
    dist_mat[:,0] = np.ravel(np.abs(pseudo_dist))

    for ii in xrange(1, k):
        # Calculate probability
        min_dist = np.min(dist_mat[:,0:ii], axis=1)
        prob = np.power(min_dist, 2)
        prob = prob/sum(prob)

        # Sample next mean with probability, 'prob'
        new_ind = np.random.choice(n, p=prob)
        means[ii,:] = img[new_ind,:]  # new mean

        # Update distance matrix with new mean
        pseudo_dist = get_sq_distance(img, means[ii,:])
        dist_mat[:,ii] = np.ravel(np.abs(pseudo_dist))

    print "Using Kmeans++.."
    return kmeans_driver(img, means)


def parse_args():
    """ Parse the arguements for the colour clustering driver.  Verify inputs.
    """
    usage = ("Usage: python driver.py k-number-of-output-colors "
            "{kmeans, kmeans++} /path/to/image.jpg")
    assert len(sys.argv) >= 2, "Too few arguements. " + usage

    # Determine k
    assert sys.argv[1].isdigit(), ("k-number-of-output-colors needs to be a "
                                  "digit. " + usage)
    k = int(sys.argv[1])

    # Determine algorithm type (optional input)
    try:
        assert sys.argv[2] in {"kmeans", "kmeans++"}, ("Invalid algorithm "
                                                       "type. " + usage)
        algorithm_type = sys.argv[2]
    except IndexError:
        print "No algorithm type given in args; using default algorithm 'kmeans'"
        algorithm_type = "kmeans"
   
    # Determine image path (optional input)
    try:
        img_path = sys.argv[3]
    except IndexError:
        print "No image path given in args; using default image './park.jpg'"
        img_path = "./park.jpg"

    return k, algorithm_type, img_path


def main():
    # Read arguements
    k, algor_type, img_path = parse_args()

    # Read image
    img = np.asarray(Image.open(img_path))
    n_row, n_col, n_dim = np.shape(img)
    img = img.reshape((n_row*n_col, n_dim))

    # k-clustering.  with Means? Medians? k-means++?
    # recall that medians are more robust to outliers
    cluster_num, means = kmeans_pp(img, k) if algor_type=="kmeans++" else kmeans(img,k)
    new_img = means[cluster_num]

    # Display original image
    img = Image.fromarray(img.reshape((n_row, n_col, n_dim)))
    img.show()
    
    # Display new image
    new_img = new_img.reshape((n_row, n_col, n_dim))
    new_img = Image.fromarray(new_img)
    new_img.show()

if __name__ == "__main__":
    main()
