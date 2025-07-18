from datetime import date
import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
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
     ],
     "label":[1,1,1,1]
})

#Train the model
@st.cache_resource

def train_model(data):
    data = data.dropna(subset=['text','label'])
    data['clean_text'] = data['text'].apply(clean_text)
    data = data[data['clean_text'].str.strip()!= ""]
   
    X_train, X_test, y_train, y_test = train_test_split(
       data['clean_text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
   )

    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf',MultinomialNB())
     ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

#The Streamlit UI
st.title("Spam Email Classifier")
st.markdown("Sort email dataset")

data = load_data()

# Plot of distribution- bar chart
fig, ax = plt.subplots()
sns.countplot(data=data, x='label', palette='viridis', ax=ax)
ax.set_title('Email class Distribution')
ax.set_xlabel('Label (Ham = 0, Spam = 1)')
ax.set_ylabel('Count')

st.pyplot(fig)

model, acc = train_model(data)

st.success(f" Model is trained with accuracy of: {acc:.2f}")

st.subheader("Classification Report")

#pie chart
st.subheader("Email class Proportions")

label_counts = data['label'].value_counts()
labels  = ['Ham(0)', 'Spam(1)']
sizes = [label_counts[0], label_counts[1]]
colors = ["#4000ff", "#f50a0a"]

fig2,ax2 = plt.subplots()
ax2.pie(
    sizes,
    labels = labels,
    colors = colors,
     autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12}
)
ax2.axis('equal')
st.pyplot(fig2)


# Classifying the dataset
data['clean_text'] = data['text'].apply(clean_text)
data = data[data['clean_text'].str.strip() != ""]
data['prediction'] = model.predict(data['clean_text'])

spam_emails = data[data['prediction'] != 1]
ham_emails = data[data['prediction'] != 0]

st.subheader("Spam Emails")
st.write(spam_emails[['text']].reset_index(drop=True))

st.subheader("Legitimate Emails")
st.write(ham_emails[['text']].reset_index(drop=True))

# Sidebar for custom email prediction
st.sidebar.header("Try a Custom Email")
user_input = st.sidebar.text_area("Enter Email text here")
if user_input:
    cleaned = clean_text(user_input)
    prediction = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]

    st.sidebar.write(f"prediction: **{'Spam' if prediction == 1 else 'Legitimate'}**")
    st.sidebar.write(f"Confidence- Legitimate: {proba[0]:.2f}, Spam: {proba[1]:.2f}")
    

