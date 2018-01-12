# colour_clustering
Display an image with k-representative colours.  (k < number of colours in original image)

# Usage
```
python driver.py <k-output-colours> {kmeans, kmeans++} /path/to/image.jpg
```
example: `python driver.py 6 kmeans++ ./park.jpg`


Basic usage with default inputs.  Will default to 'kmeans' and './park.jpg'.
```
python driver.py <k-output-colours>
```
example: `python driver.py 6`

# Example
In the park picture below, we can see that the trees are very brightly coloured and distinct from the rest of the scenary.  With k=6, we see that kmeans colour clustering does not really pick out the trees, where as kmeans++ still captures that distinction.  Also note that kmeans took 99 loops until convergence, while kmeans++ took 53 loops.
![park](https://github.com/m3ller/colour_clustering/blob/master/park.jpg)
![kmeans](https://github.com/m3ller/colour_clustering/blob/master/park_kmeans_99.png)
![kmeanspp](https://github.com/m3ller/colour_clustering/blob/master/park_kmeanspp_53.png)
