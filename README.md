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

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/progga004/image-mosaic-using-homography.git
cd image-mosaic-using-homography
