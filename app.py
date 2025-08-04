import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Sample data (make sure you have your actual data here)
sequences = [
    "ATCGATCGATCG",
    "GCTAGCTAGCTA",
    # Add more DNA sequences for training
]
labels = [0, 1]  # 0 = Normal, 1 = Cancer

def get_kmers(sequence, k=6):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

kmer_sequences = [' '.join(get_kmers(seq)) for seq in sequences]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(kmer_sequences)

model = RandomForestClassifier()
model.fit(X, labels)

st.title("🧬 DNA Cancer Cell Identifier (Simulated Chip)")

user_input = st.text_input("🧬 Enter DNA sequence from sample (A, T, C, G only):")

if st.button("Check"):
    if not user_input or len(user_input) < 6:
        st.warning("Please enter a valid DNA sequence (at least 6 characters).")
    else:
        try:
            user_kmers = ' '.join(get_kmers(user_input.upper()))
            user_X = vectorizer.transform([user_kmers])

            prediction = model.predict(user_X)[0]
            proba = model.predict_proba(user_X)[0]
            confidence = round(max(proba) * 100, 2)

            if prediction == 1:
                st.error(f"⚠️ This DNA sample is predicted to be from a **CANCER** cell. (Confidence: {confidence}%)")
            else:
                st.success(f"✅ This DNA sample is predicted to be **NORMAL**. (Confidence: {confidence}%)")

            # Plotting the probability bar chart
            labels = ['Normal', 'Cancer']
            fig, ax = plt.subplots()
            ax.bar(labels, proba, color=['green', 'red'])
            ax.set_ylabel('Probability')
            ax.set_ylim([0, 1])
            st.pyplot(fig)

        except Exception as e:
            st.warning("Something went wrong. Make sure your input only contains A, T, C, G.")
            st.error(f"Error details: {e}")
