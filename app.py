import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample training data for the machine
training_data = {
    'email':[
        "Your account has been suspended, click here to verify",
        "Free monet waiting for you, claim now",
        "Update your bank information to avoid closure",
        "Lets schedule our team meeting for you next week",
        "Dont forget your appointment at 2 PM",
        "Lunch with the cline tomorrow",
        "Important: Review your security settings now",
        "Urgent: Your email has been comprimised",
        "Here's the monthy report your requested",
        "Reminedr: Complete your training session"
    ],
    'label':[
        'phishing', 'phishing', 'phishing',
        'safe', 'safe', 'safe',
        'phishing', 'phishing', 'safe', 'safe'
    
    ]
}
df = pd.DataFrame(training_data)

# Model Pipeline
model=Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])
model.fit(df['email'], df['label'])

# UI for the page
st.title("Email Phishing detector")
st.markdown("Enter multiple emails to classify them as **safe** or **phishing**. ")

user_inpt= st.text_area("Paste your email(s) here:", height=200)
submit = st.button("Check Emails")

if submit:
    if not user_inpt.strip():
        st.warning("Please enter at least one email")
    else:
        emails =[line.strip() for line in user_inpt.split('\n') if line.strip()]
        safe_emails =[]
        phishing_emails = []

        for idx, email in enumerate(emails, start=1):
            label = model.predict([email])[0]
            result = f"{idx}. {email}"
            if label == 'safe':
                safe_emails.append(result)
            else:
                phishing_emails.append(result)

        if safe_emails:
            st.subheader("Safe Emails")
            for item in safe_emails:
                st.markdown(f"-{item}")


        if phishing_emails:
            st.subheader("Phishing Emails")
            for item in phishing_emails:
                st.markdown(f"-{item}")        

