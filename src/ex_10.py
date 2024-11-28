import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simpsons Dialogue Sentiment Analysis", page_icon=":doughnut:")

st.title("Sentiment Analysis of Simpsons Dialogues")

st.write("This app analyzes the sentiment expressed in dialogues from The Simpsons.")

df = pd.read_csv(r'data/sentiments.csv', names=['dialogue', 'sentiment'])


sentiment_counts = df['sentiment'].value_counts()

fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
ax.set_title("Distribution of Sentiments in Simpsons Dialogues")

st.pyplot(fig)


st.subheader("Data Preview")
st.dataframe(df)