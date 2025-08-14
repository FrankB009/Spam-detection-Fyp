from datetime import date
import streamlit as st
import pandas as pd
import re
import string
import time
import os
import platform
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns


#Download the NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# load and cache the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("emails.csv")
    # Normalise the column names
    df.columns = [c.lower() for c in df.columns]
    # Rename for consistency
    if 'label' not in df.columns:
        df.rename(columns={df.columns[-1]: 'label'}, inplace=True)
    if 'text' not in df.columns:
        df.rename(columns={df.columns[0]: 'text'}, inplace=True)
    
    label_map = {"ham": 0, "spam": 1}
    if df['label'].dtype == object:
        df['label'] = df['label'].str.lower().map(label_map)  

    return df[['text', 'label']] 

    
# Preprocessing the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return " ".join(filtered)

#inject generic spam
extra_spam = pd.DataFrame({
         "text":[
         "You've won a $1,000 gift card! Click here to claim your prize.",
         "Verify your PayPal account now or it will be suspended!",
         "URGENT: Update your bank information to avoid closure.",
         "Congratulations! You have been selected as a lottery winner.",
         "Claim your free iPhone now! Just click the link and enter your details."
     ],
     "label":[1,1,1,1,1]
})

#Train the model
@st.cache_resource
def train_models(data):
    data = data.dropna(subset=['text', 'label'])
    data['clean_text'] = data['text'].apply(clean_text)
    data = data[data['clean_text'].str.strip() != ""]

    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], data['label'], test_size = 0.2, random_state=42, stratify=data['label'] 
    )

    results = []

    for name, model in [
     ("MultinomialNB", Pipeline([('tfidf',TfidfVectorizer(stop_words='english')), ('clf', MultinomialNB())])),
     ("BernoulliNB", Pipeline([('tfidf', TfidfVectorizer(stop_words='english', binary=True)), ('clf', BernoulliNB())])),
     ("MLPClassifier", Pipeline([('tfidf', TfidfVectorizer(stop_words ='english')), ('clf', MLPClassifier(hidden_layer_sizes=(100), max_iter=300, random_state=42))]))
]:
     start_time= time.time()
     cpu_start = os.times().user

     model.fit(X_train, y_train)

     cpu_end = os.times().user
     wall_time = time.time() - start_time

     y_pred = model.predict(X_test)
     acc = accuracy_score(y_test, y_pred)
     prec = precision_score(y_test, y_pred)
     rec = recall_score(y_test, y_pred)
     f1 = f1_score(y_test, y_pred)

     results.append({
        'name': name,
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'cpu_time': cpu_end - cpu_start,
        'wall_time': wall_time
    })

    return results, X_test, y_test

#The Streamlit UI
st.title("Spam Email Classifier")

data = load_data()

st.subheader("Data Summary")

total_emails = data.shape[0]
unique_texts = data['text'].nunique()

spam_messages = data[data['label']== 1]['text']
most_common_spam = spam_messages.value_counts().idxmax()
most_common_spam_count = spam_messages.value_counts().max()

st.markdown(f"- **Total emails in dataset**: {total_emails}")
st.markdown(f"- **Total unique email texts**: {unique_texts}")
st.markdown(f"- Most frequent spam message: \n\n> {most_common_spam}")



results, X_test, y_test = train_models(data)

model_mnb = next(r['model'] for r in results if r['name'] == "MultinomialNB")
model_bnb = next(r['model'] for r in results if r['name'] == "BernoulliNB")
model_mlp = next(r['model']for r in results if r['name'] == "MLPClassifier")


#Bar chart
st.subheader("Email Distribution: Bar Chart")

st.markdown("### Bar Chart")
fig, ax = plt.subplots(figsize =(8,5))
sns.countplot(data=data, x='label', palette = 'viridis', ax = ax)
ax.set_title('Email Class Distribution')
ax.set_xlabel('Label(Ham = 0, Spam = 1)')
ax.set_ylabel('Count')
st.pyplot(fig)

#Histograms
st.subheader("Text Legth Distribution")

data['text_length'] = data['text'].apply(lambda x: len(str(x)))

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Text Length Distribution**")
    fig_all, ax_all = plt.subplots()
    sns.histplot(data['text_length'], bins=30, kde=True, ax = ax_all,color='steelblue')
    ax_all.set_title("All Emails")
    ax_all.set_xlabel("Text Length")
    ax_all.set_ylabel("Frequency")
    st.pyplot(fig_all)

with col2:
    st.markdown("**Spam vs Ham Text Length**")
    fig_split, ax_split= plt.subplots()
    sns.histplot(data=data, x='text_length', hue='label', bins=30, kde=True, palette={0: 'blue', 1: 'red'}, ax=ax_split)
    ax_split.set_title("Spam vs Ham")
    ax_split.set_xlabel("Text Length")
    ax_split.set_ylabel("Frequency")
    st.pyplot(fig_split)

#Accuracy comparison
st.subheader("Model comparison")

results_df = pd.DataFrame(results)
results_df_display = results_df[["name", "accuracy","precision", "recall", "f1", "cpu_time", "wall_time"]].copy()
results_df_display.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "CPU Time","Wall Time (s)"]

st.dataframe(results_df_display.style.format({
    "Accuracy":"{:.4f}",
    "Precision":"{:.4f}",
    "Recall":"{:.4f}",
    "F1 Score":"{:.4f}",
    "CPU Time": "{:.4f}",
    "Wall Time":"{:.4f}"
}))


# Classifying the dataset
data['clean_text'] = data['text'].apply(clean_text)
data = data[data['clean_text'].str.strip() != ""]
data['prediction'] = model_mlp.predict(data['clean_text'])

spam_emails = data[data['prediction'] != 1]
ham_emails = data[data['prediction'] != 0]

#Confusion matrix
st.subheader("Confusion matrix for Models")

y_pred_mnb = model_mnb.predict(X_test)
y_pred_bnb = model_bnb.predict(X_test)
y_pred_mlp = model_mlp.predict(X_test)

cm_mnb = confusion_matrix(y_test, y_pred_mnb)
cm_bnb = confusion_matrix(y_test, y_pred_mnb)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

#confusion matrix
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**MultinomialNB**")
    fig1, ax1 = plt.subplots()
    ConfusionMatrixDisplay(cm_mnb, display_labels=["Ham", "Spam"]).plot(ax = ax1, cmap='Blues', values_format='d')
    st.pyplot(fig1)

with col2:
    st.markdown("**BernoulliNB**")
    fig2, ax2 = plt.subplots()
    ConfusionMatrixDisplay(cm_bnb, display_labels=["Ham", "Spam"]).plot(ax = ax2, cmap='Greens', values_format='d')
    st.pyplot(fig2)

with col3:
    st.markdown("**MLPClassifier**")
    fig3, ax3 = plt.subplots()
    ConfusionMatrixDisplay(cm_mlp, display_labels=["Ham", "Spam"]).plot(ax = ax3, cmap='Oranges', values_format='d')
    st.pyplot(fig3)

#Matrix comparison table

st.subheader("Dataset separation using MLPClassifier")

#MLPClassifier example tables
st.subheader("Legitimate Emails")
st.write(spam_emails[['text']].reset_index(drop=True))

st.subheader("Spam Emails")
st.write(ham_emails[['text']].reset_index(drop=True))

# Sidebar for custom email prediction
st.sidebar.header("Try a Custom Email")
user_input = st.sidebar.text_area("Enter Email text here")

model_choice = st.sidebar.selectbox("Choose a model", ["Multinomial", "Bernoulli", "MLP"])

if user_input:
    cleaned = clean_text(user_input)
    if model_choice == "Multinomial":
     prediction = model_mnb.predict([cleaned])[0]
     proba = model_mnb.predict_proba([cleaned])[0]
    elif model_choice == "Bernoulli":
        prediction = model_bnb.predict([cleaned])[0]
        proba = model_bnb.predict_proba([cleaned])[0]
    else:
        prediction = model_mlp.predict([cleaned])[0]
        proba = model_mlp.predict_proba([cleaned])[0]

    st.sidebar.markdown(f"prediction: **{'Spam' if prediction == 1 else 'Legitimate'}**")
    st.sidebar.markdown(f"Confidence:")
    st.sidebar.markdown(f"- Legitimate: `{proba[0]:.2f}`")
    st.sidebar.markdown(f"- Spam: `{proba[1]:.2f}`")

