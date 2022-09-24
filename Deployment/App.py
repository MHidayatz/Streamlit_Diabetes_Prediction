#Core pkg
import streamlit as st
import streamlit.components.v1 as stc
st.set_page_config(page_title="Text Summarization", page_icon=":tada:", layout="wide")
from streamlit_option_menu import option_menu

# LexRank Algo
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#Import NLTK
import nltk
nltk.download('punkt')

# Eda Pkg
import pandas as pd

# Text Cleaning Pkgs
import neattext as nt
import neattext.functions as nfx

# Utils
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

# Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import altair as alt
import seaborn as sns

# FIle Processing
import docx2txt
from PyPDF2 import PdfFileReader
import pdfplumber

# External Utils
from App_NLP_ultis import *

# Fxn to get Wordcloud
from wordcloud import WordCloud
def plot_wordcloud(my_text):
    my_wordcloud = WordCloud().generate(my_text)
    fig = plt.figure()
    plt.imshow(my_wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)

# Fxn to Download Result
def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "nlp_result_{}_.csv".format(timestr)
    st.markdown("### ** 📩 ⬇️ Download CSV file **")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'
    st.markdown(href, unsafe_allow_html=True)

# Fxn for LexRank
def sumy_summarizer(docx,num=2):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,num)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

# Evaluate Summary
from rouge import Rouge
def evaluate_summary(summary,reference):
    r = Rouge()
    eval_score = r.get_scores(summary,reference)
    eval_score_df = pd.DataFrame(eval_score[0])
    return eval_score_df

def main():
    st.title("NLP App with Streamlit")
    menu = ["Home", "NLP (files)", "About"]
    #choice = st.sidebar.selectbox("Menu",menu)

    choice = option_menu(
        menu_title = None,
        options = menu,
        icons = ["house", "book", "envelope"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal"
    )

    if choice =="Home":
        st.subheader("Home: Analyze Text")
        raw_text = st.text_area("Enter Text Here")
        num_of_most_common = st.sidebar.number_input("Most Common Tokens", 5, 15)
        if st.button("Analyze"):
            with st.expander("Original Text"):
                st.write(raw_text)

            with st.expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)
                

            with st.expander("Entities"):
                #entity_result = get_entities(raw_text)
                #st.write(entity_result)

                entity_result = render_entities(raw_text)
                stc.html(entity_result, height=1000,scrolling=True)

            # Layout
            col1, col2 = st.columns(2)

            with col1:
                with st.expander("Word Stats"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.expander("Top Keywords"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    st.write(keywords)
                
                with st.expander("Sentiment"):
                    sent_result = get_sentiment(raw_text)
                    st.write(sent_result)


            with col2:
                with st.expander("Plot Word Freq"):
                    fig = plt.figure()
                    top_keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    plt.bar(keywords.keys(), top_keywords.values())
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with st.expander("Plot Part of Speech"):
                    try:
                        fig = plt.figure()
                        sns.countplot(token_result_df["PoS"])
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    except:
                        st.warning("Insufficient Data: Must be more than 2")

                with st.expander("Plot Wordcloud"):
                    plot_wordcloud(raw_text)


            c1,c2 = st.columns(2)

            with c1:
                with st.expander("LexRank Summary"):
                    my_summary = sumy_summarizer(raw_text)
                    document_len = {"Original": len(raw_text), "Summary":len(my_summary)}
                    document_len_df = pd.DataFrame([document_len])
                    st.dataframe(document_len_df.T)

                    st.write(my_summary)

                    # Plot 
                    st.info("Rouge Score")
                    eval_df = evaluate_summary(my_summary, raw_text)
                    st.dataframe(eval_df.T)
                    
                    #Plot Evaluation summary
                    eval_df['metrics'] = eval_df.index
                    c=alt.Chart(eval_df).mark_bar().encode(x='metrics',y='rouge-1')
                    st.altair_chart(c)


            with c2:
                with st.expander("TextRank Summary"):
                    pass
                    #my_summary = summarize(raw_text)
                    #st.write(my_summary)


            with st.expander("Download Text Analysis Result"):
                make_downloadable(token_result_df)

    elif choice == "NLP (files)":
        st.subheader("NLP Task")

        text_file = st.file_uploader("Upload Files", type=["pdf", "docx", "txt"])
        num_of_most_common = st.sidebar.number_input("Most Common Tokens", 5, 15)

        if text_file is not None:
            if text_file.type == "application/pdf":
                raw_text = read_pdf(text_file)
                # st.write(raw_text)
            elif text_file.type == "text/plain":
                # st.write(text_file.read()) # read as bytes
                raw_text = str(text_file.read(), "utf-8")
                # st.write(raw_text)
            else:
                raw_text = docx2txt.proc

            #Copy paste from above
            with st.expander("Original Text"):
                st.write(raw_text)

            with st.expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)
                

            with st.expander("Entities"):
                #entity_result = get_entities(raw_text)
                #st.write(entity_result)

                entity_result = render_entities(raw_text)
                stc.html(entity_result, height=1000,scrolling=True)

            # Layout
            col1, col2 = st.columns(2)

            with col1:
                with st.expander("Word Stats"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.expander("Top Keywords"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    st.write(keywords)
                
                with st.expander("Sentiment"):
                    sent_result = get_sentiment(raw_text)
                    st.write(sent_result)


            with col2:
                with st.expander("Plot Word Freq"):
                    fig = plt.figure()
                    top_keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    plt.bar(keywords.keys(), top_keywords.values())
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with st.expander("Plot Part of Speech"):
                    try:
                        fig = plt.figure()
                        sns.countplot(token_result_df["PoS"])
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    except:
                        st.warning("Insufficient Data: Must be more than 2")

                with st.expander("Plot Wordcloud"):
                    plot_wordcloud(raw_text)


            c1,c2 = st.columns(2)

            with c1:
                with st.expander("LexRank Summary"):
                    my_summary = sumy_summarizer(raw_text)
                    document_len = {"Original": len(raw_text), "Summary":len(my_summary)}
                    document_len_df = pd.DataFrame([document_len])
                    st.dataframe(document_len_df.T)

                    st.write(my_summary)

                    # Plot 
                    st.info("Rouge Score")
                    eval_df = evaluate_summary(my_summary, raw_text)
                    st.dataframe(eval_df.T)
                    
                    #Plot Evaluation summary
                    eval_df['metrics'] = eval_df.index
                    c=alt.Chart(eval_df).mark_bar().encode(x='metrics',y='rouge-1')
                    st.altair_chart(c)


            with c2:
                with st.expander("TextRank Summary"):
                    pass
                    #my_summary = summarize(raw_text)
                    #st.write(my_summary)


            with st.expander("Download Text Analysis Result"):
                make_downloadable(token_result_df)

    else:
        st.subheader("About")
        st.text("Created by me, Md. Hidayat")
        st.text("Deploying Streamlit UI for NLP.")

if __name__ == '__main__':
	main()