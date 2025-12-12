# Local Curvature Analysis and Visualization Tool for Nanoparticles

This repository contains a Python tool that analyzes 3D points on the surface of nanoparticles, fits a quadratic surface patch to estimate local curvature parameters, and visualizes the results through 3D plots and histograms.

## Overview

The tool performs the following tasks:

* **Point Extraction:** Obtains the x, y, and z coordinates of points on the surface of a nanoparticle.
* **Visualization:** Plots the generated points in 3D space.
* **Curvature Estimation:** For each point, defines a local patch (based on a distance threshold) and fits a quadratic surface. It then computes curvature parameters (principal curvatures, Gaussian curvature, and mean curvature).
* **Histogram & Color Mapping:** Generates visual histograms (in both bar and line formats) and 3D color bar plots for Gaussian curvature (KG) and mean curvature (Km).

## Requirements

* Python 3.x
* NumPy
* Matplotlib
* SciPy

## Output

When you run the script, it will generate several files in the working directory, including:

* **points.jpeg** – A 3D plot of the points.
* **points.txt** – A text file with point coordinates.
* **curvature_param.txt** – A text file with curvature parameters.
* **Histogram Images:** e.g., `KG_hist_bar.jpeg`, `KG_hist_line.jpeg`, `Km_hist_bar.jpeg`, `Km_hist_line.jpeg`.
* **Additional Visualizations:** e.g., `R_squared.jpeg` and color bar plots for curvature.

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements, please feel free to submit a pull request or open an issue.

## Author

Hassan Ghermezcheshme
