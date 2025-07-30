from datetime import date
import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
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

#Multinomial Naive Bayes
    model_mnb = Pipeline([
     ('tfidf', TfidfVectorizer(stop_words ='english')),
     ('clf', MultinomialNB())
 ])
    model_mnb.fit(X_train, y_train)
    acc_mnb = accuracy_score(y_test, model_mnb.predict(X_test))


#Bernoulli Naive Bayes
    model_bnb = Pipeline([
     ('tfidf', TfidfVectorizer(stop_words ='english', binary=True)),
     ('clf', BernoulliNB())

 ])
    
    model_bnb.fit(X_train, y_train)
    acc_bnb = accuracy_score(y_test, model_bnb.predict(X_test))

#MLP Classifier 
    model_mlp = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf' , MLPClassifier(hidden_layer_sizes=(100,),max_iter=300,random_state=42))
])
    model_mlp.fit(X_train,y_train)
    acc_mlp = accuracy_score(y_test, model_mlp.predict(X_test))

    return model_mnb, model_bnb, model_mlp, acc_mnb, acc_bnb,  acc_mlp

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



model_mnb, model_bnb, model_mlp, acc_mnb, acc_bnb, acc_mlp =train_models(data)

# Model accuracy
st.header("Model Accuracy Comparison")
st.markdown(f"- **Multinomial Naive Bayes Accuracy**: `{acc_mnb:.4f}`")
st.markdown(f"- **Bernoulli Naive Bayes Accuracy**: `{acc_bnb:.4f}`")
st.markdown(f"- **MLP Classifier Accuracy**: `{acc_mlp:.4f}` ")

#Bar and Pie charts
st.subheader("Email Distribution: Bar and pie Chart")

col1,col2 = st.columns(2)

with col1:
    st.markdown("### Bar Chart")
    fig, ax = plt.subplots(figsize =(5,5))
    sns.countplot(data=data, x='label', palette = 'viridis', ax = ax)
    ax.set_title('Email Class Distribution')
    ax.set_xlabel('Label(Ham = 0, Spam = 1)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

with col2:
    st.markdown("### Pie Chart")
    label_counts = data['label'].value_counts()
    labels = ['Ham(0)', 'Spam(1)']
    sizes = [label_counts[0], label_counts[1]]
    colors = ["#002AFF", "#ff0000"]

    fig2, ax2 = plt.subplots(figsize = (5,5))
    ax2.pie(
    sizes,
    labels = labels,
    colors = colors,
    autopct ='%1.1f%%',
    startangle = 90,
    textprops = {'fontsize': 12}
)

    ax2.axis('equal')
    st.pyplot(fig2)


# Classifying the dataset
data['clean_text'] = data['text'].apply(clean_text)
data = data[data['clean_text'].str.strip() != ""]
data['prediction'] = model_mnb.predict(data['clean_text'])

spam_emails = data[data['prediction'] != 1]
ham_emails = data[data['prediction'] != 0]

st.subheader("Spam Emails")
st.write(spam_emails[['text']].reset_index(drop=True))

st.subheader("Legitimate Emails")
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

