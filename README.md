# ML Model Deployment Homework

This project fulfills the requirements for the regression and classification assignment. It includes:
1.  **Regression:** Predicts California Housing prices using three different PyTorch neural networks.
2.  **Classification:** Predicts the presence of heart disease using three different PyTorch neural networks.

All 6 models were trained in a separate Python notebook and exported to the `.onnx` format. This web application loads the `.onnx` files and performs live inference in the browser using `onnxruntime-web`.

## Project Files

* `index.html`: The main HTML structure, including forms and input validation.
* `style.css`: The minimal, dark-themed, responsive stylesheet.
* `app.js`: The JavaScript logic that loads models, preprocesses input, and runs predictions.
* `ort.min.js`: The (optional) local copy of the ONNX runtime library.
* `*.onnx`: The 6 exported PyTorch models (3 for regression, 3 for classification).

## 🚀 How to Run Locally

You cannot just double-click `index.html` due to browser security policies (CORS). You must run it from a local server.

**Option 1: Using Python (Easiest)**
1.  Open a terminal in this project folder.
2.  Run the following command:
    ```bash
    python3 -m http.server
    ```
    (Use `python -m SimpleHTTPServer` if you have an older Python version).
3.  Open your browser and go to: **`http://localhost:8000`**

**Option 2: Using VS Code**
1.  Install the "Live Server" extension from the VS Code marketplace.
2.  Right-click on `index.html` in the file explorer.
3.  Click "Open with Live Server".

## Deployment

This site is deployed on GitHub Pages. The input fields have been validated with `min` and `max` attributes to prevent out-of-distribution (OOD) data, which can cause nonsensical predictions.
