## How to use

 - Download .src/stitch.py to local repository
 - run python3 stitch.py [imagepath]
 - The output panorama image is generated in the same folder

## How to creat panorama

 - Using SIFT algorithm to extract features from two images, calculate the cosine similarity between features from different images. Find a certain number of best-matching pairs according to their cosine similariy
 - Use the matching points to construct a Homographical matrix that represents the transformation from one image to another.
 - Use the homographical matrix to transform feature points in the test set. Use RANSAC to select different combinations of feature points and find the best homographical matrix.
