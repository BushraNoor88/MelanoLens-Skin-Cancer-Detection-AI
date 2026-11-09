"""CNN modules for MelanoLens AI (U-Net segmentation + MobileNetV2 classification).

Training happens OUTSIDE Flask (python -m cnn.segmentation.train / cnn.classification.train).
Flask only loads the saved models for inference. TensorFlow is imported lazily, so the classical
pipeline keeps working even when TensorFlow is not installed.
"""
