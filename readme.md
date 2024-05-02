# CSL7360: Computer Vision - Assignment 2
This repository contains the implementation of two image segmentation techniques: K-means clustering and Spectral clustering (Radio-Cut based clustering). The report B20AI062.pdf provides a detailed description of the implementations, experiments, and results.

## Introduction
Image segmentation is a fundamental task in computer vision, aiming to partition an image into multiple segments or regions based on certain similarities or characteristics. This assignment explores two popular segmentation techniques: K-means clustering and Spectral clustering.
### K-means Segmentation
The K-means segmentation is implemented from scratch. It takes an image and the number of clusters k as input. The function iteratively assigns each pixel to the nearest cluster centroid and updates the centroids based on the assigned pixels. The segmented image is obtained by replacing each pixel with its corresponding cluster centroid value.
### Radio-Cut Spectral Clustering
The Radio-Cut Spectral Clustering implementation constructs a graph from the image pixels using the image.img_to_graph function from the scikit-image library. The Laplacian matrix L is computed from the graph's adjacency matrix A. The eigenvectors of the Laplacian matrix are then calculated, and K-means clustering is performed on these eigenvectors to partition the graph into clusters. The resulting cluster assignments are stored in the label_im array.
## Experiments and Results
The report includes experiments performed on two input images, using both K-means clustering and Spectral clustering with k=3 and k=6 clusters. The results are presented as segmented images.
### Result Discussion
The report discusses the performance and observations of the two segmentation techniques:

- K-means Clustering:

    For k=3, the algorithm tends to segment images based on broad color regions, often merging visually distinct objects or regions.
    For k=6, the algorithm captures more detailed segmentation, separating objects and regions with slightly different color shades.
    However, K-means struggles with capturing non-convex cluster shapes and can lead to over-segmentation or under-segmentation in certain cases.


- Radio-Cut Spectral Clustering:

    Spectral clustering generally provides better segmentation results, capturing the natural boundaries and shapes of objects or regions more accurately.
    For k=3, the algorithm effectively separates the main objects or regions in the images, even when they have similar colors but distinct boundaries.
    For k=6, the algorithm further refines the segmentation, separating smaller details and textures within the main objects or regions.
    Spectral clustering is particularly effective in handling non-convex cluster shapes and preserving the overall structure of the image.

## Replication of Results

To replicate the results you need to first create and activate a virtual environment by running the following commands 


```
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

After that you can run our python notebook.

The report also discusses the reasoning for poor results in Spectral clustering, which is attributed to the loss of prominent information during image resizing.

## Conclusion
The report concludes that while K-means clustering is computationally efficient and suitable for simple segmentation tasks, Spectral clustering provides more accurate and detailed segmentation results, especially for complex images with non-convex cluster shapes and varying textures. However, Spectral clustering comes at a higher computational cost compared to the simpler K-means algorithm.