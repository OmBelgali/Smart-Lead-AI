# SmartLeadAI – Intelligent Lead Qualification Tool

### 🧠 Overview

SmartLeadAI is a lightweight AI-powered tool that predicts the quality of business leads (High / Medium / Low).
It demonstrates how machine learning can automate and optimize the **lead generation process** by helping companies prioritize high-value prospects.

---

### 🎯 Objective

Businesses often collect thousands of leads but struggle to identify which ones are worth pursuing.
SmartLeadAI uses a **Random Forest Classifier** to analyze company data such as:

* Industry
* Employees
* Funding
* Website Traffic
* Email Response Rate

and outputs a **Lead Score** to help teams focus on top opportunities.

---

### ⚙️ How It Works

1. **Data Collection:** Uses `leads.csv` containing company information.
2. **Data Preprocessing:** Cleans and encodes features.
3. **Model Training:** Trains a RandomForest model to learn lead quality patterns.
4. **Prediction:** Predicts lead potential for new companies using `predict_lead.py`.
5. **(Optional)** Visualizes top leads using Matplotlib.

---

### 🧩 File Structure

```
SmartLeadAI/
│
├── leads.csv                  # Dataset of company leads
├── templates                  # Web Page
    ├── index.html
├── static                      
    ├── style.css   
├── smartleadai.py             # Model training script
├── predict_lead.py            # Prediction script
├── requirements.txt           # Dependencies
├── app.py                     
├── Visualization
└── README.md                  # Documentation
```

---

### 🚀 How to Run the Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/SmartLeadAI.git
   cd SmartLeadAI
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**

   ```bash
   python smartleadai.py
   ```

4. **Make predictions**

   ```bash
   python predict_lead.py
   ```

---

### 📊 Example Output

```
Accuracy: 0.88
Predicted Lead Quality: High
```

---

### 🎥 Video Walkthrough

📹 [Watch Project Demo](https://drive.google.com/file/d/1g09wFh1FE7RjYob2nfLANKNpS6y58EPy/view?usp=sharing)

---

### 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Joblib
* Matplotlib (optional)

---

### 🏢 Business Relevance

This project aligns with **Caprae Capital’s mission** to use AI for business transformation.
SmartLeadAI demonstrates how machine learning can enhance **lead generation**, **sales efficiency**, and **strategic decision-making** across industries.

---

### 👤 Author

**Om Belgali**
Machine Learning Engineer Intern (Caprae Capital – Handbook Submission)
📧 [[om.belgali@gmail.com]

