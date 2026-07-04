# 🛍️ Enterprise Retail Intelligence & Recommendation Suite

An end-to-end data engineering and analytics web application that transforms raw e-commerce transaction logs into actionable business insights. This production-ready system handles automated customer segmentation and serves real-time product recommendations.

## 🔗 Live Application
👉 https://online-retail-analytics-2nnnnnn8.streamlit.app/

## 🛠️ System Architecture & Workflow
1. **Data Pipeline (`train_pipeline.py`)**: Cleans 540,000+ transaction rows, filters out returns, aggregates customer behaviors, and compresses artifacts to prevent cloud runtime memory crashes.
2. **Customer Segmentation**: Utilizes standardized **RFM (Recency, Frequency, Monetary)** features fed into an optimized **K-Means Clustering** algorithm to map buyers into distinct behavioral segments (High-Value, Regular, Occasional, At-Risk).
3. **Recommendation Engine**: Implements a sparsified **Cosine Similarity** user-item interaction matrix, pre-cached into a memory-efficient Top-K product correlation mapping.
4. **Interactive Dashboard (`app.py`)**: A Streamlit frontend displaying live business metrics, dynamic affinity heatmaps, and a 3D customer cluster visualization space.

## 📦 Tech Stack & Core Dependencies
* **Frontend UI:** Streamlit, HTML5/CSS Injection
* **Data Visualizations:** Plotly Express (Interactive 3D Scatter & Matrix Heatmaps)
* **Machine Learning & Engineering:** Scikit-Learn (KMeans, StandardScaler, Cosine Similarity)
* **Data Processing:** Pandas, NumPy, Gzip Compression
