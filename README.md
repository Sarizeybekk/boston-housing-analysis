# 🏠 Boston Housing Analysis - Streamlit App  

This project is a **Streamlit web application** for **Boston Housing Dataset** analysis.  
It includes **data preprocessing, missing value handling, visualizations, and insights.**  

Developed as part of the **Turkcell Geleceği Yazan Kadınlar** program, and **still being improved.** 🚀  

## 📊 Features  
- **Handles missing values** using mean, median, and KNN imputation  
- **Creates insightful visualizations** (scatter plots, heatmaps, etc.)  
- **Provides an interactive UI** with Streamlit  
- **Allows downloading the cleaned dataset**  

## 🚀 Deployment  
The application is live on **Hugging Face Spaces!** Try it here:  

👉 **[Boston Housing Analysis - Hugging Face](https://huggingface.co/spaces/sarizeybek/boston-housing-analysis)**  

## 📂 Files  
- `boston_housing_app.py` → Streamlit app  
- `HousingData.csv` → Raw dataset  
- `Boston_Housing_Cleaned.csv` → Processed dataset  
- `.gitignore` → Excludes unnecessary files  

## 🔧 Installation & Run Locally  
To run the project locally:  
```bash
git clone https://github.com/Sarizeybekk/boston-housing-analysis.git
cd boston-housing-analysis
pip install -r requirements.txt
streamlit run boston_housing_app.py
