# Main program to run colour clustering
from PIL import Image
import numpy as np


""" Find k-representative colours of the image
"""
def kmeans(img, k):
    # Randomly pick k pixels as initial cluster "means"
    # Random indices are picked without replacement; to avoid duplicate means
    n = len(img) 
    rand_ind = np.random.choice(n, size=k, replace=False) 
    means = img[rand_ind, :]

    # For each cluster, compare distance with pixel.  We'll use Euclidean distance
    #TODO: Use while-loop and monitor with convergence
    for ii in xrange(1):
        # Calculating mean inner-product
        mean_prod = np.power(means,2)
        mean_prod = np.sum(mean_prod, axis=1)
        mean_prod = np.expand_dims(mean_prod, axis=0)

        # Calculating sample/mean inner-product
        pseudo_dist = np.matmul(img, means.T)   # (n,d) x (d, k)
        pseudo_dist = pseudo_dist + mean_prod
        
        # Find smallest distance 
        cluster_num = np.argmin(pseudo_dist, axis=1)
    # Return cluster number, cluster mean
    return cluster_num, means

def main():
    # Read image
    #img_path = raw_input().strip()
    img_path = "./tetris_confetti.jpg"
    img = np.asarray(Image.open(img_path))
    n_row, n_col, n_dim = np.shape(img)
    img = img.reshape((n_row*n_col, n_dim))

    # k-clustering.  with Means? Medians? k-means++?
    # recall that medians are more robust to outliers
    cluster_num, means = kmeans(img, 100)
    new_img = means[cluster_num]
    new_img = new_img.reshape((n_row, n_col, n_dim))
    
    # Display new image
    new_img = Image.fromarray(new_img)
    new_img.show()

    # Display original image
    img = Image.fromarray(img.reshape((n_row, n_col, n_dim)))
    img.show()

if __name__ == "__main__":
    main()
