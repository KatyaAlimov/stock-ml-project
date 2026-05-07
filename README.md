## 📈 Project Overview
This project focuses on predicting short-term financial shifts (5 trading days into the future) using daily historical OHLCV data. By leveraging a regression-based approach, we aim to quantify future profitability decimal returns to provide strategic decision support for minimizing risk and maximizing yields.

## 📊 Dataset
We utilized a dataset sourced from **[Kaggle](https://www.kaggle.com/datasets/asadullahcreative/us-stock-market-historical-ohlcv-dataset/data)** containing **184,138 records** of US stock market data.
* **Features:** Open, High, Low, Close, and Volume.
* **Engineering:** Includes technical indicators such as SMA_5, Volatility (rolling std), and Price Change.
* **Processing:** 80/20 chronological train-test split with Standard Scaling and outlier clipping (±20%).

## 🧠 Machine Learning Models

## 🏆 Key Results
| Model | Mean Squared Error (MSE) | R² Score |
| :--- | :--- | :--- |
| Linear Regression | 0.0039 | 0.012 |
| KNN Regression | 0.0036 | 0.082 |
| **Neural Network** | **0.0032** | **0.138** |

The Neural Network provided the best performance, capturing complex interactions between volume and volatility that linear models missed.

## 🚀 Future Roadmap
* **Hyperparameter Tuning:** Implementing Grid Search for model optimization.
* **Ensemble Methods:** Exploring Random Forests and Boosting techniques.
* **Dimensionality Reduction:** Using PCA to filter market noise.

## 🛠️ Installation & Usage
1. Clone the repository: `git clone https://github.com/KatyaAlimov/stock-ml-project.git`
2. Install dependencies: `pip install pandas numpy matplotlib tensorflow scikit-learn kagglehub`
3. Run the analysis: `python main.py`
