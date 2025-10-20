This repository contains a comprehensive research project and accompanying implementation that explores the relationship between false positive exoplanet detections and binary star systems using machine learning techniques. The core research question investigates whether false positive exoplanet data actually contains binary star systems. False positives in exoplanet detection waste valuable telescope time and resources, and this study aims to determine what percentage of these false positives can be attributed to eclipsing binary stars.

The project employs a two-stage machine learning approach: first using a K-Nearest Neighbors (KNN) model to identify exoplanets and generate false positive data, then applying a combined Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN) model to classify binary star systems from these false positives.

The research yielded significant insights into astronomical classification. The KNN model achieved 88% accuracy with an F1 score of 85% for exoplanet detection. When the false positives from this model were analyzed using the LSTM-CNN architecture, approximately 6% were identified as binary star systems, with the binary classification model achieving 80% accuracy. This suggests that while some false positives are indeed binary stars, the majority likely represent other celestial phenomena or detection artifacts.

## Project Structure

### Binary Star Classification (`binarystars.ipynb`)
A Jupyter Notebook implementing a GRU (Gated Recurrent Unit) neural network for classifying binary star systems. This notebook focuses on processing stellar parameters including orbital period, Kepler magnitude, morphology, temperature, and short cadence indicators. The model architecture features multiple GRU layers with dropout regularization and achieves robust performance on binary star classification tasks.

### Exoplanet Detection (`exoplanet_detection.ipynb`)
An advanced deep learning implementation for exoplanet detection from astronomical time-series data. This notebook explores multiple neural network architectures including standard CNNs, hybrid CNN-LSTM models, and advanced CNN architectures with skip connections. The implementation incorporates sophisticated signal processing techniques such as Fourier transforms and various filtering methods to enhance detection accuracy.

## Technical Approach

The project demonstrates several advanced machine learning techniques in astronomical contexts. For exoplanet detection, multiple dimensionality reduction methods were employed to handle the high-dimensional flux data, while the binary star classification leveraged temporal pattern recognition capabilities of LSTMs combined with feature extraction power of CNNs. Both implementations include comprehensive data preprocessing, feature engineering, and model evaluation protocols.

## Applications and Implications

This research has practical applications for astronomical data analysis pipelines. By improving the differentiation between genuine exoplanet transits and binary star eclipses, the models can help optimize telescope time allocation and reduce false positive rates in exoplanet surveys. The methodologies developed could also be extended to classify other types of variable astronomical phenomena.

## Future Work

The research paper identifies several directions for future improvement, including incorporating additional observational data such as wavelength measurements and Doppler shift analysis, exploring alternative model architectures like GRU or RNN networks, and expanding the classification scope to include other celestial bodies that exhibit transit-like signatures.

## Implementation Details

All projects were developed using Python with standard scientific computing libraries (NumPy, Pandas, Scikit-learn) and deep learning frameworks (TensorFlow/Keras). The code is designed to be modular and extensible, allowing researchers to build upon the existing implementations for related astronomical classification tasks.

This repository represents a significant contribution to the application of machine learning in astronomy, providing both theoretical insights and practical implementations for distinguishing between different types of celestial transits.
