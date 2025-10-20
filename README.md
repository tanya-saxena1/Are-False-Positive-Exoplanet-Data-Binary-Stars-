## Introduction

This repository contains a comprehensive research project and accompanying implementation that explores the relationship between false positive exoplanet detections and binary star systems using machine learning techniques. The core research question investigates whether false positive exoplanet data actually contains binary star systems. False positives in exoplanet detection waste valuable telescope time and resources, and this study aims to determine what percentage of these false positives can be attributed to eclipsing binary stars.

The project employs a two-stage machine learning approach: first using a K-Nearest Neighbors (KNN) model to identify exoplanets and generate false positive data, then applying a combined Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN) model to classify binary star systems from these false positives.

The research yielded significant insights into astronomical classification. The KNN model achieved 88% accuracy with an F1 score of 85% for exoplanet detection. When the false positives from this model were analyzed using the LSTM-CNN architecture, approximately 6% were identified as binary star systems, with the binary classification model achieving 80% accuracy. This suggests that while some false positives are indeed binary stars, the majority likely represent other celestial phenomena or detection artifacts.

## Overview

False positives in exoplanet detection waste valuable telescope time and resources. Identifying and understanding the sources of these false positives, such as eclipsing binary stars, can lead to improved algorithms and methods, increasing the reliability of future exoplanet discoveries. Additionally, this research enriches our understanding of stellar systems, particularly eclipsing binaries. By distinguishing these systems from genuine exoplanet detections, we can contribute valuable data to stellar astrophysics, facilitating the study of star formation, evolution, and dynamics.

In this context, using machine learning would be most helpful in helping to differentiate between true false positives and eclipsing binary stars. Given the substantial amount of numerical data involved, machine learning is well-suited for efficiently sorting and analysing this information. Providing scientists with the tool for the detection of exoplanets, eclipsing binary stars, and other celestial bodies will enhance their accuracy in this field.


## Dataset
For this project, the Kepler Campaign 3 exoplanet and eclipsing binary stars datasets from the NASA archives were used and implemented. The numerical data for the exoplanet detection was the label, 1 for non-exoplanet, and 2 for exoplanet, and the flux given per orbit, labelled as “FLUX.1”, “FLUX.2” etc. The different flux is the light intensity recorded, at different points in time throughout the campaign. The numerical data for the binary stars systems was narrowed down to the orbital period (period) and its error (period_err) in days, the Kepler magnitude (kmag), the morphology parameter (morph), the effective temperature of the star (TEFF) and the short cadence indicator (SC).


## Project Structure

### Binary Star Classification (`binarystars.ipynb`)
A Jupyter Notebook implementing a GRU (Gated Recurrent Unit) neural network for classifying binary star systems. This notebook focuses on processing stellar parameters including orbital period, Kepler magnitude, morphology, temperature, and short cadence indicators. The model architecture features multiple GRU layers with dropout regularization and achieves robust performance on binary star classification tasks.

### Exoplanet Detection (`exoplanet_detection.ipynb`)
An advanced deep learning implementation for exoplanet detection from astronomical time-series data. This notebook explores multiple neural network architectures including standard CNNs, hybrid CNN-LSTM models, and advanced CNN architectures with skip connections. The implementation incorporates sophisticated signal processing techniques such as Fourier transforms and various filtering methods to enhance detection accuracy.

## Technical Approach

The project demonstrates several advanced machine learning techniques in astronomical contexts. For exoplanet detection, multiple dimensionality reduction methods were employed to handle the high-dimensional flux data, while the binary star classification leveraged temporal pattern recognition capabilities of LSTMs combined with feature extraction power of CNNs. Both implementations include comprehensive data preprocessing, feature engineering, and model evaluation protocols.

## Applications and Implications

This research has practical applications for astronomical data analysis pipelines. By improving the differentiation between genuine exoplanet transits and binary star eclipses, the models can help optimize telescope time allocation and reduce false positive rates in exoplanet surveys. The methodologies developed could also be extended to classify other types of variable astronomical phenomena.

## Implementation Details

All projects were developed using Python with standard scientific computing libraries (NumPy, Pandas, Scikit-learn) and deep learning frameworks (TensorFlow/Keras). The code is designed to be modular and extensible, allowing researchers to build upon the existing implementations for related astronomical classification tasks.

This repository represents a significant contribution to the application of machine learning in astronomy, providing both theoretical insights and practical implementations for distinguishing between different types of celestial transits.


## Future Work

The research paper identifies several directions for future improvement, including incorporating additional observational data such as wavelength measurements and Doppler shift analysis, exploring alternative model architectures like GRU or RNN networks, and expanding the classification scope to include other celestial bodies that exhibit transit-like signatures.

## Acknowledgments

I would like to thank **InSpirit AI** for giving me the opportunity and platform to develop this work through their InSpirit AI + X 1:1 mentorship program as well as **Andrei Isichenko** for his invaluable guidance, support, and mentorship throughout this research project. 
