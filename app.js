// Wait for the DOM and ONNX runtime to be ready
document.addEventListener('DOMContentLoaded', () => {
    // --- SCALER VALUES FROM PYTHON NOTEBOOK ---
    
    // From California Housing (8 features)
    const REGRESSION_MEANS = [3.8807542575097025, 28.60828488372093, 5.435235020487511, 1.0966847487895384, 1426.453003875969, 3.096961194668754, 35.64314922480603, -119.58229045542558];
    const REGRESSION_STDS = [1.904236257747077, 12.602117730095111, 2.387302579723397, 0.4332014262644869, 1137.0219480466546, 11.578393509846865, 2.136600595041553, 2.005592806566646];
    
    // From Heart Disease (5 numeric features)
    const CLASSIFICATION_NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'];
    const CLASSIFICATION_MEANS = [54.61181434599156, 132.25316455696202, 248.45991561181435, 148.48101265822785, 1.0379746835443038];
    const CLASSIFICATION_STDS = [8.963660089125463, 17.999157285310883, 52.29552784803182, 23.18201366629552, 1.1343472032046225];
    
    // One-Hot-Encoder categories in the *exact* order from the Python ColumnTransformer
    const CLASSIFICATION_CATEGORICAL_INFO = {
        'sex': [0, 1],
        'cp': [1, 2, 3, 4],
        'fbs': [0, 1],
        'restecg': [0, 1, 2],
        'exang': [0, 1],
        'slope': [1, 2, 3],
        'ca': [0, 1, 2, 3],
        'thal': [3, 6, 7]
    };
    
    // --- End of Scaler Values ---

    // Get all the DOM elements
    const regForm = document.getElementById('regression-form');
    const regModelSelect = document.getElementById('regression-model');
    const regResultDiv = document.getElementById('regression-result');
    
    const classForm = document.getElementById('classification-form');
    const classModelSelect = document.getElementById('classification-model');
    const classResultDiv = document.getElementById('classification-result');

    const models = {}; // To store pre-loaded models

    // Helper function to load a model
    async function loadModel(modelPath) {
        try {
            // NOTE: This MUST load from your GitHub pages URL.
            // To run locally, you MUST change this to:
            // const baseUrl = ""; 
            // and run a local python server
            const baseUrl = "https://nsk246.github.io/ml-classification-regression-app/";
            return await ort.InferenceSession.create(baseUrl + modelPath);
        } catch (e) {
            console.error(`Error loading model ${modelPath}: ${e}`);
            return null;
        }
    }

    // *** UPDATED TO PRE-LOAD ALL 14 MODELS ***
    Promise.all([
        // My Regression Models (4)
        loadModel('regression_simple.onnx').then(m => models['regression_simple.onnx'] = m),
        loadModel('regression_deep.onnx').then(m => models['regression_deep.onnx'] = m),
        loadModel('regression_optimized.onnx').then(m => models['regression_optimized.onnx'] = m),
        loadModel('regression_xgboost.onnx').then(m => models['regression_xgboost.onnx'] = m), 
        
        // Prof's Regression Models (4)
        loadModel('regression_linregnet.onnx').then(m => models['regression_linregnet.onnx'] = m),
        loadModel('regression_mlp_net_reg.onnx').then(m => models['regression_mlp_net_reg.onnx'] = m),
        loadModel('regression_dl_net_reg.onnx').then(m => models['regression_dl_net_reg.onnx'] = m),
        loadModel('regression_linearplusnonlinear_net.onnx').then(m => models['regression_linearplusnonlinear_net.onnx'] = m),
        
        // My Classification Models (3)
        loadModel('classification_simple.onnx').then(m => models['classification_simple.onnx'] = m),
        loadModel('classification_deep.onnx').then(m => models['classification_deep.onnx'] = m),
        loadModel('classification_optimized.onnx').then(m => models['classification_optimized.onnx'] = m),
        
        // Prof's Classification Models (2)
        loadModel('classification_mlp_net_class.onnx').then(m => models['classification_mlp_net_class.onnx'] = m),
        loadModel('classification_dl_net_class.onnx').then(m => models['classification_dl_net_class.onnx'] = m),


    ]).then(() => {
        console.log("All 13 models loaded successfully.");
    });

    // 1. --- Handle Regression Form Submission ---
    regForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Stop the form from reloading
        
       
        try {
            const modelName = regModelSelect.value;
            const session = models[modelName];
            if (!session) {
                alert("Model is not loaded yet, please wait.");
                return;
            }

            // Get and scale the 8 input features
            const inputIds = ['medinc', 'houseage', 'averooms', 'avebedrms', 'population', 'aveoccup', 'latitude', 'longitude'];
            const scaledData = new Float32Array(8);
            
            for (let i = 0; i < inputIds.length; i++) {
                const val = parseFloat(document.getElementById(inputIds[i]).value);
                scaledData[i] = (val - REGRESSION_MEANS[i]) / REGRESSION_STDS[i];
            }
            
            const inputTensor = new ort.Tensor('float32', scaledData, [1, 8]);
            const feeds = { 'features': inputTensor }; 
            
            const results = await session.run(feeds);
            
            // This is the debug line. It will now stay in the console.
            console.log("Regression results object:", results); 
            
            // This is the line that is likely failing.
            const predictionData = results.prediction || results.output_label || results.variable;
            const prediction = predictionData.data[0]; 
            
            const formattedPrice = (prediction * 100000).toLocaleString('en-US', {
                style: 'currency',
                currency: 'USD',
                maximumFractionDigits: 0
            });
            
            regResultDiv.innerHTML = `Predicted Median Value: <strong>${formattedPrice}</strong>`;
            regResultDiv.className = 'result-box result-regression'; 
            regResultDiv.style.display = 'block'; 

        } catch (err) {
            console.error("Prediction failed!", err);
            // Display the error on the page so we can see it
            regResultDiv.innerHTML = `<strong>Prediction Error:</strong> ${err.message}. Check console for details.`;
            regResultDiv.className = 'result-box result-positive'; // Make it red
            regResultDiv.style.display = 'block'; 
        }
        
    });

    // 2. --- Handle Classification Form Submission ---
    classForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        try {
            const modelName = classModelSelect.value;
            const session = models[modelName];
            if (!session) {
                alert("Model is not loaded yet, please wait.");
                return;
            }
            
            const processedData = preprocessClassificationInput();
            
            const inputTensor = new ort.Tensor('float32', processedData, [1, 28]);
            const feeds = { 'features': inputTensor };
            
            const results = await session.run(feeds);
            
            let logits;
            if (results.logits) {
                logits = results.logits.data; // Get logits from PyTorch models
            } else {
                // Reconstruct logits from XGBoost probabilities
                const prob_class_0 = results.output_probability.data[0];
                const prob_class_1 = results.output_probability.data[1];
                logits = [Math.log(prob_class_0 + 1e-9), Math.log(prob_class_1 + 1e-9)]; 
            }
            
            const probabilities = softmax(logits);
            const prob_class_0 = probabilities[0]; // Probability of "Negative"
            const prob_class_1 = probabilities[1]; // Probability of "Positive"
            
            const prediction = prob_class_1 > prob_class_0 ? 1 : 0;
            
            if (prediction === 1) {
                const percentage = (prob_class_1 * 100).toFixed(1);
                classResultDiv.innerHTML = `Prediction: <strong>Positive for Heart Disease</strong><br><small>Confidence: ${percentage}%</small>`;
                classResultDiv.className = 'result-box result-positive';
            } else {
                const percentage = (prob_class_0 * 100).toFixed(1);
                classResultDiv.innerHTML = `Prediction: <strong>Negative for Heart Disease</strong><br><small>Confidence: ${percentage}%</small>`;
                classResultDiv.className = 'result-box result-negative';
            }
            classResultDiv.style.display = 'block';
        } catch (err) {
            console.error("Prediction failed!", err);
            classResultDiv.innerHTML = `<strong>Prediction Error:</strong> ${err.message}. Check console for details.`;
            classResultDiv.className = 'result-box result-positive'; // Make it red
            classResultDiv.style.display = 'block'; 
        }
    });
    
    /**
     * Replicates the Python ColumnTransformer.
     */
    function preprocessClassificationInput() {
        const output = new Float32Array(28); 
        let outputIndex = 0;

        // 1. Process 5 Numeric Features
        for (let i = 0; i < CLASSIFICATION_NUMERIC_FEATURES.length; i++) {
            const id = CLASSIFICATION_NUMERIC_FEATURES[i];
            const val = parseFloat(document.getElementById(id).value);
            output[outputIndex++] = (val - CLASSIFICATION_MEANS[i]) / CLASSIFICATION_STDS[i];
        }
        
        // 2. Process 8 Categorical Features (One-Hot Encoding)
        for (const [id, categories] of Object.entries(CLASSIFICATION_CATEGORICAL_INFO)) {
            const selectedValue = parseFloat(document.getElementById(id).value);
            for (const category of categories) {
                output[outputIndex++] = (category === selectedValue) ? 1.0 : 0.0;
            }
        }
        
        if (outputIndex !== 28) {
            console.error(`Preprocessor error: Expected 28 features, got ${outputIndex}`);
        }
        
        return output;
    }

    /**
     * Computes the softmax function on an array of numbers (logits).
     */
    function softmax(arr) {
        // Add a check for non-finite numbers which can come from log(0)
        const safeArr = arr.map(x => (isFinite(x) ? x : -Infinity));
        const maxLogit = Math.max(...safeArr);
        const exps = safeArr.map(x => Math.exp(x - maxLogit)); 
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => (e / sum));
    }
});
