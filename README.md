# Image Mosaic Creation Using Corner Detection, Feature Matching, and Homography

## Overview

This project demonstrates the process of creating an image mosaic using advanced computer vision techniques, including corner detection, feature matching, homography matrix computation, and image warping. The goal is to align and stitch two images seamlessly into a single panoramic mosaic, utilizing the RANSAC algorithm for robust feature matching and inlier detection. The project is a comprehensive implementation of key concepts in image processing and computer vision, showcasing my proficiency in these areas.

## Features

- **Corner Detection**: Detects significant points of interest (corners) in both source and destination images.
- **Feature Matching**: Extracts and matches intensity patches around detected corners using Normalized Cross-Correlation (NCC) as the similarity metric.
- **Homography Matrix Computation**: Computes the homography matrix using a robust set of inliers detected through the RANSAC algorithm.
- **Image Warping**: Warps the source image into the coordinate system of the destination image using the computed homography matrix.
- **Image Mosaic Creation**: Creates a smooth mosaic of the source and destination images using feathering techniques for blending.
- **Bonus**: Implementation of image mosaic blending using feathering to achieve smooth transitions between images.

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- NumPy
- Matplotlib (for visualization)

### Descriptions

- **`__pycache__/`**: Directory for Python cache files, automatically generated.
- **`Dst_corners.jpg`**: Output image showing detected corners in the destination image.
- **`Dst_patches.jpg`**: Output image showing patches extracted from the destination image.
- **`Image1.jpg`**: Source image used for creating the mosaic.
- **`Image2.jpg`**: Destination image used for creating the mosaic.
- **`README.md`**: Documentation file that provides an overview of the project, setup instructions, usage, and other details.
- **`Src_corners.jpg`**: Output image showing detected corners in the source image.
- **`Src_patches.jpg`**: Output image showing patches extracted from the source image.
- **`cornerdetection.py`**: Contains functions for image processing, including Gaussian smoothing, gradient calculations, corner detection, and extraction.
- **`finalone.py`**: Main script that runs the entire image mosaic creation process.
- **`inlier_matches.jpg`**: Output image showing inlier matches after applying the RANSAC algorithm.
- **`matches.jpg`**: Output image showing initial matches between source and destination images.
- **`stiching.jpg`**: The final stitched image output showing the completed mosaic.

### Installation and Running the Script

Clone the repository to your local machine:

```bash
git clone https://github.com/progga004/image-mosaic-using-homography.git
cd image-mosaic-using-homography
python finalone.py



