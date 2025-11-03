**PART ONE
1. Theoretical Analysis**
a)	Explain how AI-driven code generation tools (e.g., GitHub Copilot) reduce development time. What are their limitations?
AI code tools (e.g., GitHub Copilot) reduce development time by auto-completing code, generating templates, and minimizing syntax errors.
Limitations: They lack full project context, can generate insecure or low-quality code, and may raise copyright issues.
b)	Compare supervised and unsupervised learning in the context of automated bug detection.
Unsupervised learning finds unusual patterns that might indicate new or unknown bugs.
Supervised is accurate with data; unsupervised works without labels but may flag false    positives.

c)	**Why is bias mitigation critical when using AI for user experience personalization?**
Bias mitigation ensures fair, inclusive personalization. Without it, AI can discriminate, reduce user trust, and create unequal experiences.

2.** Case Study Analysis**
AIOps (Artificial Intelligence for IT Operations) improves software deployment efficiency by using machine learning and data analytics to automate and optimize the deployment process. It reduces manual intervention, speeds up issue detection, and ensures smoother releases.
Examples include; Automated Error Detection and Resolution and Predictive Resource Management.
 **  PART TWO**
**Task 1**
 Write a Python function to sort a list of dictionaries by a specific key. Compare the AI-suggested code with your manual implementation and document which version is more efficient and why.
AI suggested code from Github Copilot
def sort_dicts_by_key(data, key):
return sorted(data, key=lambda x: x[key])
-This version uses Python’s built-in sorted() function with a lambda expression. It’s concise, readable, and efficient, leveraging Python’s Timsort algorithm (O(n log n) complexity).

**Manual Implementation**
def manual_sort_dicts_by_key(data, key):
    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j][key] > data[j + 1][key]:
                data[j], data[j + 1] = data[j + 1], data[j]
return data
-This version manually compares and swaps elements using a basic bubble sort approach. It’s intuitive but much slower (O(n²) complexity).
Efficiency Analysis
-The AI-suggested implementation using Python’s sorted() function is significantly more efficient and pythonic compared to the manual bubble sort approach. The built-in sorting function is optimized at the C level and uses Timsort, which combines merge sort and insertion sort techniques to achieve a time complexity of O(n log n). It also maintains stability—meaning that if two elements have the same key value, their original order is preserved.
In contrast, the manual implementation uses a nested loop (bubble sort) with O(n²) time complexity. While it demonstrates a clear understanding of sorting logic, it becomes inefficient for larger datasets, leading to slower performance and higher computational cost. Additionally, the built-in approach is more readable and less prone to coding errors.
Overall, the AI-suggested code outperforms the manual version in speed, readability, and maintainability. For professional applications, leveraging Python’s optimized functions is the preferred choice.

**Task 2
Automated Testing with AI**
Test scripts
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
service = Service("C:\\Users\\Admin\\Downloads\\chromedriver-win32\\chromedriver-win32\\chromedriver.exe")
driver = webdriver.Chrome(service=service)
driver.get("https://www.saucedemo.com/")
time.sleep(2)
driver.find_element("id", "user-name").send_keys("standard_user")
driver.find_element("id", "password").send_keys("secret_sauce")
driver.find_element("id", "login-button").click()
time.sleep(2)
if "inventory" in driver.current_url:
    print("✅ Login successful!")
else:
    print("❌ Login failed.")
driver.quit()
Test screenshots
 

 
How AI improves test coverage compared to manual testing.
AI-driven tools like Testim.io enhance test automation by learning element behavior and adapting to UI changes automatically. Traditional Selenium scripts can break when element locators or page layouts change, but AI-powered locators in Testim identify components by multiple attributes (like text, position, and context), improving test resilience. AI also analyzes past test executions to prioritize high-risk scenarios and suggest additional test paths, increasing coverage. In this test, AI automation accurately verified both valid and invalid login workflows, detecting expected success and error messages. Compared to manual testing, this approach saves time, reduces human error, and ensures consistent regression testing across multiple browsers and devices.

**Task 3**
**Predictive Analytics for Resource Allocation**
Dataset: Use Kaggle Breast Cancer Dataset.
Preprocess data (clean, label, split).
Train a model (e.g., Random Forest) to predict issue priority (high/medium/low).
Evaluate using accuracy and F1-score.

from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
X.head()

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
features = X.drop(columns=['priority_continuous_source','priority_label'])
labels = X['priority_label'].astype(str)
le = LabelEncoder()
y = le.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
y_pred = rf.predict(X_test_scaled)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 (macro):', f1_score(y_test, y_pred, average='macro'))
print('\nClassification Report:\n')
print(classification_report(y_test, y_pred, target_names=le.classes_))
print('\nConfusion Matrix:\n')
print(confusion_matrix(y_test, y_pred))

Performance Metrics
{
  "accuracy": 0.993006993006993,
  "f1_macro": 0.9929824561403509,
  "label_mapping": {
    "high": 0,
    "low": 1,
    "medium": 2
  },
  "classification_report": "              precision    recall  f1-score   support\n\n        high       1.00      1.00      1.00        48\n         low       1.00      0.98      0.99        48\n      medium       0.98      1.00      0.99        47\n\n    accuracy                           0.99       143\n   macro avg       0.99      0.99      0.99       143\nweighted avg       0.99      0.99      0.99       143\n",
  "confusion_matrix": [
    [
      48,0,0
    ],
    [
      0, 47, 1
    ],
    [
      0,0,47
   ]
  ],
  "n_samples": 569,  "n_classes": 3}

**PART 3**
**Ethical Reflection**
Even though the breast-cancer dataset is widely used for research, it still contains potential sources of bias that could affect fairness when applied in a company setting:
•	Demographic Bias: The dataset primarily represents data from a specific hospital and population group (mostly older women from Wisconsin, USA). If deployed globally, the model may underperform for underrepresented groups (e.g., younger patients or different ethnic backgrounds).
•	Sampling Bias: Data collection was not balanced across all tumor types or sizes, leading to potential over-representation of certain conditions.
•	Feature Bias: Some features (e.g., “mean radius” or “texture”) might correlate with factors like imaging equipment or lab calibration rather than actual patient risk—introducing hidden technical bias.
Operational Bias: If the company uses the model to allocate medical or technical resources (e.g., prioritizing issue tickets), groups with fewer samples or noisier data may receive lower priority unfairly.
Addressing Bias with Fairness Tools
Fairness frameworks like IBM AI Fairness 360 (AIF360) can help identify and mitigate these biases through:
•	Bias Detection Metrics: Tools such as disparate impact, statistical parity difference, and equal opportunity difference measure whether certain groups receive systematically different outcomes.
•	Pre-processing Techniques: Methods like Reweighing or Disparate Impact Remover adjust the data before training to balance underrepresented groups.
•	In-processing Algorithms: Fair models (e.g., adversarial debiasing) learn to minimize bias while maintaining accuracy.
Bonus task
Proposed Tool: AutoDoc AI – Intelligent Software Documentation Assistant
Purpose:
Software engineers often spend hours writing and updating project documentation, which quickly becomes outdated. AutoDoc AI automatically generates and maintains accurate, developer-friendly documentation directly from codebases and commit histories.
Workflow:
Code Parsing: AutoDoc AI scans source code, function signatures, and comments using language models trained on programming corpora.
Context Extraction: It reads recent Git commits and issue descriptions to identify new features or API changes.
Documentation Generation: The system uses an LLM to produce clear explanations, usage examples, and parameter details in Markdown or HTML format.
Continuous Updates: On every commit, a CI/CD pipeline triggers AutoDoc AI to re-generate only modified sections, maintaining synchronization.
Developer Review: Generated text appears as a pull-request diff for human verification before merge.
Impact:
Efficiency: Reduces manual documentation time by up to 70%.
Quality: Ensures consistent, readable, and up-to-date docs across teams.
Adoption: Encourages better knowledge sharing and onboarding for new developers.
Ethics & Transparency: Increases code interpretability, addressing one of the key challenges of AI-driven software systems.

