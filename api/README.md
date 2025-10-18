# AI Clean Code Analysis API

FastAPI backend for the AI Clean Code Analysis Tool IntelliJ plugin.

## Setup

1. **Install dependencies:**
   ```bash
   cd api
   pip install -r requirements.txt
   ```

2. **Ensure trained models are available:**
   - Make sure you have trained CNN models in `training/checkpoints/`
   - Required files:
     - `training/checkpoints/methods_vocab.json`
     - `training/checkpoints/methods_textcnn.pt`
     - `training/checkpoints/classes_vocab.json`
     - `training/checkpoints/classes_textcnn.pt`

3. **Ensure datasets are available:**
   - `dataset/official_dataset_methods.csv`
   - `dataset/official_dataset_classes.csv`

## Running the API

From the project root directory:

```bash
python -m api.main
```

Or with uvicorn directly:

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### `POST /predict/method`

Predicts code quality for a Java method.

**Request body:**
```json
{
  "code_snippet": "public void someMethod() { ... }"
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "good",
  "confidence": 0.85
}
```

Prediction values:
- `0` = "good" (Green)
- `1` = "changes_recommended" (Yellow)
- `2` = "changes_required" (Red)

### `POST /predict/class`

Predicts code quality for a Java class.

**Request body:**
```json
{
  "code_snippet": "public class SomeClass { ... }",
  "average_method_score": 1.2
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "changes_recommended",
  "confidence": 0.75
}
```

The `average_method_score` is optional and calculated automatically by the plugin based on method predictions.

## How it Works

The API uses an ensemble approach combining:

1. **Heuristics** (10% weight) - Rule-based analysis of code metrics
2. **KNN Classifier** (50% weight) - K-Nearest Neighbors with precomputed features
3. **CNN Model** (40% weight) - Deep learning on tokenized code

The final prediction is a weighted combination of all three components.

## Integration with IntelliJ Plugin

The IntelliJ plugin:
1. Extracts all methods and classes from the currently open Java file
2. Sends each method to `/predict/method` endpoint
3. Calculates average method score
4. Sends each class to `/predict/class` endpoint with the average method score
5. Highlights code based on predictions:
   - Green = Good code
   - Yellow = Changes recommended
   - Red = Changes required

## Testing the API

You can test the API using curl:

```bash
# Test method prediction
curl -X POST "http://localhost:8000/predict/method" \
  -H "Content-Type: application/json" \
  -d '{"code_snippet": "public void test() { System.out.println(\"Hello\"); }"}'

# Test class prediction
curl -X POST "http://localhost:8000/predict/class" \
  -H "Content-Type: application/json" \
  -d '{"code_snippet": "public class Test { }", "average_method_score": 1.0}'
```

Or visit `http://localhost:8000/docs` for interactive API documentation.

## Troubleshooting

- **Models not found:** Run the training scripts first to generate model checkpoints
- **Port already in use:** Change the port in `main.py` or kill the process using port 8000
- **Import errors:** Make sure you're running from the project root directory
