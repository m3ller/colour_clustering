# Main program to run colour clustering
from PIL import Image
import numpy as np
import sys


""" Find k-representative colours of the image
"""

#TODO: could introduce random restarts
#TODO: introduce time-its
def kmeans_driver(img, k, means):
    #TODO: make sure means are in float32
    # For each cluster, compare distance with pixel.  We'll use Euclidean distance
    old_cluster_num = np.random.randint(k, size=len(img))
    similarity = 0
    loop_counter = 0    # Counts number of loops needed before reaching convergence

    while similarity < 0.999:
        # Calculating distance; note that 'means' needs to be converted to float for calculations
        # Calculating mean inner-product
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

        
        # Monitor convergence.  See if the clustering has changed
        similarity = np.sum(old_cluster_num == cluster_num) / float(len(cluster_num))
        old_cluster_num = cluster_num
        loop_counter += 1
  
    print("Number of loops needed before reaching convergence: {0}".format(loop_counter))
    means = (np.round(means)).astype(np.uint8)
    return cluster_num, means

def kmeans(img, k):
    # Randomly pick k pixels as initial cluster "means"
    # Random indices are picked without replacement; to avoid duplicate means
    n = len(img) 
    rand_ind = np.random.choice(n, size=k, replace=False) 
    means = img[rand_ind, :].astype(np.float32) 
    return kmeans_driver(img, k, means)

def kmeans_pp(img, k):
    n = len(img)
    means = np.zeros((k, 3), dtype=np.float32)
    rand_ind = np.random.randint(n)
    means[0,:]= img[rand_ind,:].astype(np.float32)

    # Pick means based on a probability distribution
    dist_mat = np.inf * np.ones((n, k))
    pseudo_dist = np.sum(np.power(img.astype(np.float32), 2), axis=1) -2*np.matmul(img, means[0,:].T) + np.sum(np.power(means[0,:],2))
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
        pseudo_dist = np.sum(np.power(img.astype(np.float32), 2), axis=1) -2*np.matmul(img, means[ii,:].T) + np.sum(np.power(means[ii,:], 2))
        dist_mat[:,ii] = np.abs(pseudo_dist)

    return kmeans_driver(img, k, means)

def main():
    # Read arguements
    #TODO: option to pick between kmeans and kmeans++
    # Usage: python driver.py k-output-colors /path/to/image.jpg
    assert len(sys.argv) >= 2, "Usage: python driver.py k-number-of-output-colors /path/to/image.jpg"
    k = int(sys.argv[1])

    try:
        img_path = sys.argv[2]
    except IndexError:
        print "No image path given in args; using default image './lego.jpg'"
        img_path = "./lego.jpg"

    try:
        function = sys.argv[3]
    except IndexError:
        print "No function given in args; using default function 'kmeans'"
        function = "kmeans"

    # Read image
    img = np.asarray(Image.open(img_path))
    n_row, n_col, n_dim = np.shape(img)
    img = img.reshape((n_row*n_col, n_dim))

    # k-clustering.  with Means? Medians? k-means++?
    # recall that medians are more robust to outliers
    cluster_num, means = kmeans_pp(img, k) if function=="kmeans++" else kmeans(img,k)
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
