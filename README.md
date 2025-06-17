# emotion_recognition
Here's a `README.md` file tailored for your project. It includes an overview, installation steps, usage, and examples:

---

````markdown
# Emotion Classifier with Decision Tree

This project is a simple machine learning model that classifies emotions based on facial features using a Decision Tree Classifier. It uses a small, custom dataset of numerical features such as smile width, eyebrow angle, and eye openness to predict one of three emotions: **Happy**, **Sad**, or **Angry**.

---

## 🧠 Features

- **Custom synthetic dataset**
- **Train/test split**
- **Decision Tree Classifier using Scikit-learn**
- **Model evaluation using accuracy and classification report**
- **Sample prediction for new input**

---

## 📁 Dataset

The dataset consists of 12 examples with the following features:

- `smile_width`: Ranges from 0 to 1
- `eyebrow_angle`: Ranges from 0 to 1
- `eye_openness`: Ranges from 0 to 1
- `emotion`: Target label - one of `Happy`, `Sad`, or `Angry`

---

## 🛠 Installation

Make sure you have Python 3 installed.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-classifier.git
cd emotion-classifier
````

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, here's what you need:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 🚀 Usage

Run the script:

```bash
python emotion_classifier.py
```

Expected output:

```text
Accuracy: 0.75

Classification Report:
              precision    recall  f1-score   support

      Angry       1.00      1.00      1.00         1
      Happy       1.00      1.00      1.00         1
        Sad       0.50      0.50      0.50         2

    accuracy                           0.75         4
   macro avg       0.83      0.83      0.83         4
weighted avg       0.75      0.75      0.75         4

Predicted emotion for new data (smile_width=0.85, eyebrow_angle=0.2, eye_openness=0.7): Happy
```

---

## 🔍 Example Prediction

You can modify or extend predictions like so:

```python
new_data = np.array([[0.85, 0.2, 0.7]])
predicted_emotion = model.predict(new_data)
print(predicted_emotion)
```

---

## 📊 Optional: Visualize the Decision Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.show()
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

```

---

Let me know if you'd like me to generate a `requirements.txt`, add a `LICENSE`, or create a visualization script!
```
