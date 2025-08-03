import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Sample DNA data
dna_data = [
    ("ATCGTACGATCG", 1),
    ("CGATCGTACGAT", 0),
    ("TACGATCGATAC", 1),
    ("GATCGATACGAT", 0),
    ("ATCGATCGTACG", 1),
    ("CGTACGATCGAT", 0),
    ("TACGTACGTACG", 1),
    ("CGTACGTACGTA", 0),
    ("GCTAGCTAGCTA", 1),
    ("TAGCTAGCTAGC", 0),
    ("CGTACGATACGA", 1),
    ("TACGATACGTAG", 0),
]

sequences = [seq for seq, label in dna_data]
labels = [label for seq, label in dna_data]

def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

kmer_sequences = [' '.join(get_kmers(seq)) for seq in sequences]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(kmer_sequences)

model = RandomForestClassifier()
model.fit(X, labels)

st.title("🧬 DNA Cancer Cell Identifier (Simulated Chip)")

user_input = st.text_input("🔬 Enter DNA sequence from sample (A, T, C, G only):")

if st.button("Check"):
    if not user_input or len(user_input) < 6:
        st.warning("Please enter a valid DNA sequence (at least 6 characters).")
    else:
        try:
            user_kmers = ' '.join(get_kmers(user_input.upper()))
            user_X = vectorizer.transform([user_kmers])
            prediction = model.predict(user_X)[0]

            if prediction == 1:
                st.error("⚠️ This DNA sample is predicted to be from a **CANCER** cell.")
            else:
                st.success("✅ This DNA sample is predicted to be **NORMAL**.")
        except:
            st.warning("Something went wrong. Make sure your input only contains A, T, C, G.")
