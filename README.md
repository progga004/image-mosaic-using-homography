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
- Matplotlib (optional, for visualization)
### Descriptions

- **`images/`**: This directory contains input images, intermediate output images such as detected corners, patches, and the final stitched mosaic image.
- **`finalone.py`**: This is the main script that runs the entire process of image mosaic creation, including reading images, detecting corners, extracting patches, matching features, computing homography, and stitching the images together.
- **`cornerdetection.py`**: A module that provides functions for processing images, including Gaussian smoothing, gradient calculations, corner detection, and extraction.
- **`README.md`**: The documentation file that provides an overview of the project, setup instructions, usage, and other details.

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/progga004/image-mosaic-using-homography.git
cd image-mosaic-using-homography



