# colour_clustering
Display an image with k-representative colours.  (k < number of colours in original image)

## Usage
```
python driver.py <k-output-colours> {kmeans, kmeans++} /path/to/image.jpg
```
example: `python driver.py 6 kmeans++ ./park.jpg`


Basic usage with default inputs.  Will default to 'kmeans' and './park.jpg'.
```
python driver.py <k-output-colours>
```
example: `python driver.py 6`

## Example 1
In the park picture below, we can see that the trees are very brightly coloured and distinct from the rest of the scenary.  

With k=6, we see that Kmeans colour clustering does not really distinguish the trees from the buildings, where as kmeans++ is able to capture this contrast.  This is due to the image having comparatively fewer orange/red pixels (colour of the trees) to blue/grey pixels (colour of buildings, sky, water), and due to the way Kmeans and Kmeans++ initialize their means.

Since there are fewer orange/red pixels to blue/grey pixels, the uniformally random initial means selected by Kmeans is unlikely to be orange/red.  Even in further iterations, since there are so few orange/red pixels (even within their own cluster), it becomes difficult to shift the weight of their closest mean to more of the red spectrum.

In comparison, Kmeans++ gives more weight to selecting means that are significantly different from the means chosen so far.  In this way, even though the first selected mean may be from grey/blue pixels, there is a much higher chance for a orange/red mean to be selected afterwards.  Hence the distinction between the red/orange trees and everything else in Kmeans++.  

Also note that Kmeans took 99 loops until convergence, while Kmeans++ took 53 loops.   

| <img src="https://github.com/m3ller/colour_clustering/blob/master/park.jpg" width="430"> |
|:---:|
| <em> My photograph of George Wainborn Park, Vancouver, Canada. </em> |

| <img src="https://github.com/m3ller/colour_clustering/blob/master/park_kmeans_99.png" width="430"> | <img src="https://github.com/m3ller/colour_clustering/blob/master/park_kmeanspp_53.png" width="430"> |
|:---:|:---:|
| <em> Kmeans, where k=6. </em> | <em> Kmeans++, where k=6. </em> |

## Example 2
In the Lego pictures below, where k=4, we can see the Kmeans and Kmeans++ produce very similar outputs.  This is in part, due to Lego's discretize colour palette, which makes determining the colour clusters more clear-cut.

Note for the output below, Kmeans took 26 loops, while Kmeans++ took 17 loops.

| <img src="https://github.com/m3ller/colour_clustering/blob/master/lego.jpg"> |
|:---:|
| <em> My photograph of a Lego living room. </em> |

| <img src="https://github.com/m3ller/colour_clustering/blob/master/lego_kmeans_26.png"> | <img src="https://github.com/m3ller/colour_clustering/blob/master/park_kmeanspp_17.png"> |
|:---:|:---:|
| <em> Kmeans, where k=4. </em> | <em> Kmeans++, where k=4. </em> |
