# Main program to run colour clustering
from PIL import Image
import numpy as np
import sys


""" Find k-representative colours of the image
"""

def get_sq_distance(img, mean):
    """ Returns the squared distances between each pixel in the image 'img'
    and the 'mean'.

    Squared distance calculation:
        sq_dist = sum_i ( a_i - b_i )^2, where i is a dimension
                = inner_product( vec(a)-vec(b), vec(a)-vec(b) )
                = vec(a)^2 - 2*inner_prod( vec(a), vec(b) ) + vec(b)^2
    """
    a_sq = np.sum(np.power(img.astype(np.float32), 2), axis=1)
    ab = np.matmul(img, mean.T)
    b_sq = np.sum(np.power(mean,2))

    sq_dist = a_sq - 2*ab + b_sq
    return sq_dist


#TODO: Could introduce random restarts
#TODO: Compare convergence. Suspect overall distance-to-means of kmeans is high
def kmeans_driver(img, k, means):
    """ Run the Kmeans algorithm on the image 'img' with the initial means,
    'means'.
    """
    # Calculations for means must be in floats
    if means.dtype != 'float':
        means = means.astype(np.float32)

    # For each cluster, compare Euclidean distance from pixel
    old_cluster_num = np.random.randint(k, size=len(img))
    similarity = 0
    loop_counter = 0    # Counts number of loops before reaching convergence

    while similarity < 1.0:
        #TODO: add this distance to a generalized squared distance function
        # Calculating distance squared
        mean_prod = np.power(means, 2)
        mean_prod = np.sum(mean_prod, axis=1)
        mean_prod = np.expand_dims(mean_prod, axis=0)   # (1,k)

        # Calculating sample/mean inner-product
        pseudo_dist = np.matmul(img, means.T)   # (n,d) x (d,k)
        pseudo_dist = -2*pseudo_dist + mean_prod
        
        # Find smallest distance 
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
    return kmeans_driver(img, k, means)

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
    dist_mat[:,0] = np.abs(pseudo_dist)

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
        dist_mat[:,ii] = np.abs(pseudo_dist)

    return kmeans_driver(img, k, means)


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
