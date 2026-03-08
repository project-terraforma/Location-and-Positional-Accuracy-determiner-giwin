# TruePin: Spatial Repositioning Prototype

**TruePin** is a prototype project aimed at measuring and correcting spatial offset errors in open-source Points of Interest (POI) data, specifically focusing on the [Overture Maps Foundation](https://overturemaps.org/) dataset.

While POI datasets often drop coordinates in empty parking lots or near the center of massive shopping malls, TruePin seeks to mathematically reposition these pins onto the actual physical building footprints to optimize hyper-local routing and delivery navigation.

---

## 📊 Final Project Results

### 1. The Baseline Offset (Overture Error Measurement)
After building a hand-labeled Ground Truth dataset of 500 real-world locations, the spatial discrepancy (Haversine distance) between the standard Overture coordinate and the true street-facing entrance was measured:

**Global Offset Metrics:**
* **Mean Error:** 39.08 meters
* **Median Error:** 12.50 meters
* **RMSE:** 123.87 meters
* **90th Percentile:** 66.61 meters
* **95th Percentile:** 137.43 meters

**Error by Structural Category:**
* **Large Area:** Mean 78.65m / Median 41.27m
* **Mall or Nested Area:** Mean 67.88m / Median 29.12m
* **Standalone Store:** Mean 35.27m / Median 11.59m

**Error by Data Source:**
* **Meta Only:** Mean 41.69m / Median 10.90m
* **Microsoft Only:** Mean 50.95m / Median 19.20m
* **Both Sources:** Mean 23.08m / Median 10.87m

### 2. The Solution (Uber H3 Spatial Discretization)
To fix this, the infinite continuous map was discretized into groups of 19 local ~20m hexagonal grids (H3 Resolution 11) surrounding each Overture pin. The task was converted into a Machine Learning classification problem: *"Which of these 19 hexagons contains the actual TruePin?"*

Overture Maps API was used to extract **Building Footprints** and **Road Segments** surrounding each hex to engineer spatial features.

**Spatial Classification Accuracy Results:**
* **Baseline Accuracy (Leaving pin exactly where Overture put it):** `57.80%`
* **XGBoost ML Algorithm:** `61.73%`
* **Rule-Based Heuristic (Snap to nearest building):** `61.75%`

*Conclusion: Both the heuristic and XGBoost model proved that factoring in basic Building Footprints directly increases pinpoint accuracy globally by over ~4.0%.*

---

## 🚀 How to Run the Pipeline

If you clone this repository, you must run the pipeline sequentially to rebuild the data from scratch:

**0. Set Up**
* set up a python virtual environment and install the libraries in requirements.txt
* export your Gemini API Key as export GEMINI_API_KEY='your_api_key_here'

**1. Define the Ground Truth Datasets**
* run `streamlit run label_app.py` (A Streamlit application to hand-label ground truth points on a Folium Map).
* *Outputs: `ground_truth_labels.csv`*

**2. Measure Baseline Offsets & Perform Gemini Categorization**
* run `python calculate_offset.py`
* run `python categorize_and_analyze.py` (Requires `GEMINI_API_KEY` set in your environment).
* *Outputs: `ground_truth_with_errors.csv` and `ground_truth_fully_analyzed.csv`*

**3. Build the Spatial Hexagonal Model**
* run `python build_h3_grid.py` (Discretizes map space via Uber's H3 library).
* run `python feature_engineering.py` (Downloads live Overture building/road Polygons and calculates intersection/distance matrices).
* *Outputs: `h3_classification_features.csv`*

**4. Train the ML Models & Visualize**
* run `python train_model.py` (Trains the XGBoost model).
* run `python generate_map.py` (Creates interactive Folium visualizations of the worst offset distances).
* run `python predict_and_visualize.py` (Takes Unseen points, fetches Overture polygons, infers through the XGBoost model, and plots the new positions).
* *Outputs: `offset_map.html` & `model_prediction_map.html`*
