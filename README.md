# 🛍️ Product Categorization and Color Prediction using Machine Learning

🚀 **A Machine Learning project for predicting product categories and colors in e-commerce using NLP & CNNs.**

## 📖 Project Overview
E-commerce platforms like **Etsy** host millions of products, making **automated categorization** and **color prediction** essential for searchability and recommendations. This project leverages **Machine Learning** and **Deep Learning** techniques to classify **product categories** and predict **primary & secondary colors** based on text and image data.

### ✨ **Key Objectives:**
- **Top Category Prediction**: Using **Logistic Regression** on textual data.
- **Bottom Category Prediction**: Implemented with **Support Vector Machine (SVM)**.
- **Color Prediction**: Leveraging **Convolutional Neural Networks (CNNs)** for image classification.

🔍 **Dataset**: Includes 240,000 product records containing **titles, descriptions, and images**.  
📦 **Format**: Data stored in **Parquet** files for efficient processing.  

---

## 📊 Models Used
| Model | Task | Input Data |
|--------|----------------|----------------|
| **Logistic Regression** | Predict **Top Category ID** | Product Titles & Descriptions |
| **Support Vector Machine (SVM)** | Predict **Bottom Category ID** | Text-based Features |
| **Convolutional Neural Network (CNN)** | Predict **Primary & Secondary Colors** | Product Images |

📌 The models were **trained and optimized** to maximize **F1-score** for accuracy in classification.

---

## 📂 Dataset & Preprocessing
### **📌 Data Cleaning**
- Removed missing values (80% missing data in categorical columns).
- Standardized text descriptions and category names.
- Applied **feature selection** techniques for better model performance.

### **📌 Data Preprocessing**
- **Text Processing**: Used **CountVectorizer** and **TF-IDF Transformer** for numerical representation.
- **Feature Engineering**: Extracted relevant features using **SelectFromModel** (Linear SVC).
- **Image Normalization**: Rescaled pixel values (0-1) for CNN training.

📌 **Final dataset stored in `data/predictions_23265555.parquet`** for efficient ML processing.

---

## 🛠 Technologies Used
🔹 **Programming Language**: Python   
🔹 **Machine Learning & NLP**: Scikit-Learn, TensorFlow  
🔹 **Text Processing**: CountVectorizer, TF-IDF Transformer  
🔹 **Feature Selection & Model Optimization**: GridSearchCV, SelectFromModel  
🔹 **Data Processing**: Pandas, NumPy  
🔹 **Evaluation Metrics**: F1-Score, Accuracy  
🔹 **Visualization**: Matplotlib, Seaborn  
🔹 **Storage & Data Handling**: Parquet files for efficient dataset processing  

✅ **Jupyter Notebook** was used for model development & testing.

---

## 🚀 Running the Code
**1️⃣ Clone the Repository**
```bash
git clone https://github.com/ronitloke/Product-Categorization-and-Color-Prediction-using-Machine-Learning-techniques.git
cd Product-Categorization-and-Color-Prediction-using-Machine-Learning-techniques
```
**2️⃣ Install Required Dependencies**
```bash
pip install pandas scikit-learn tensorflow matplotlib seaborn
```
**3️⃣ Run the Jupyter Notebook**
```bash
jupyter notebook notebooks/ML_Etsy_code.ipynb
```
**4️⃣ View Predictions**
```bash
import pandas as pd
df = pd.read_parquet("data/predictions_23265555.parquet")
print(df.head())
```
---

## 📜 Report & Documentation
📄 Detailed methodology, experiments, and results are documented in:

📥 **documentation/Product Categorization and Color Prediction using Machine Learning techniques.pdf**

---

## 📌 Results & Evaluation
### ✔ Model Performance
| Model                            | Accuracy          |
|----------------------------------|------------------|
| **Logistic Regression** (Top Category) | **84.2% (F1-Score)** |
| **SVM** (Bottom Category)         | **60.8% (F1-Score)** |
| **CNN** (Primary Color Prediction) | **41.2%** |
| **CNN** (Secondary Color Prediction) | **22.5%** |

📌 **Challenges**: Color classification was difficult due to varying image conditions.

---

## 📌 Future Improvements
- Use Deep Learning-based NLP models (**BERT, GPT**) for text classification  
- Train a more robust CNN model with **ImageNet pre-trained weights**  
- Apply **multi-label classification** for more accurate color predictions  
- Enhance dataset with **more diverse samples & real-time learning mechanisms** 

---

## 💡 Contribution Guidelines
```bash
git clone https://github.com/ronitloke/Product-Categorization-and-Color-Prediction-using-Machine-Learning-techniques.git
git checkout -b feature-branch
git commit -m "Your changes"
git push origin feature-branch
```

---

## 🚀 Follow for Updates  
🌟 If you like this project, please star this repository!  
📬 Contact: ronitloke10@gmail.com

---

