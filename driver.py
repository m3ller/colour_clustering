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


    
    # Find smallest distance 
    # Return cluster number, cluster mean

    pass

def main():
    # Read image
    #img_path = raw_input().strip()
    img_path = "./tetris_confetti.jpg"
    img = np.asarray(Image.open(img_path))
    n_row, n_col, n_dim = np.shape(img)
    img = img.reshape((n_row*n_col, n_dim))

    # k-clustering.  with Means? Medians? k-means++?
    # recall that medians are more robust to outliers
    

    # Display image
    new_img = Image.fromarray(img)
    new_img.show()

if __name__ == "__main__":
    main()
