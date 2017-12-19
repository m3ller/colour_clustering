# Main program to run colour clustering
from PIL import Image
import numpy as np

def main():
    # Read image
    #img_path = raw_input().strip()
    img_path = "./tetris_confetti.jpg"
    img = np.asarray(Image.open(img_path))
    n_row, n_col, n_dim = np.shape(img)

    # k-clustering.  with Means? Medians? k-means++?

    # Display image
    new_img = Image.fromarray(img)
    new_img.show()

if __name__ == "__main__":
    main()
