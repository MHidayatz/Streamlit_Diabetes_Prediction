#Core pkg
import streamlit as st
import streamlit.components.v1 as stc
from streamlit_option_menu import option_menu

# Eda Pkg
import pandas as pd

# Load Data viz pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

# Load Data
@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df


def explore(data):
    df_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
    numerical_cols = df_types[~df_types['Data Type'].isin(['object',
               'bool'])].index.values
    df_types['Count'] = data.count()
    df_types['Unique Values'] = data.nunique()
    df_types['Min'] = data[numerical_cols].min()
    df_types['Max'] = data[numerical_cols].max()
    df_types['Average'] = data[numerical_cols].mean()
    df_types['Median'] = data[numerical_cols].median()
    df_types['St. Dev.'] = data[numerical_cols].std()
    return df_types.astype(str)

def run_eda_app():
    st.subheader("From Exploratory Data Analysis")
    #df = load_data(r"C:\Users\user\Documents\GitHub\Streamlit_Diabetes_Prediction\Data\diabetes_data_upload.csv")
    #df_encoded = load_data(r"C:\Users\user\Documents\GitHub\Streamlit_Diabetes_Prediction\Data\diabetes_data_upload_clean.csv")
    #freq_df = load_data(r"C:\Users\user\Documents\GitHub\Streamlit_Diabetes_Prediction\Data\freqdist_of_age_data.csv")
    df = load_data(r"https://raw.githubusercontent.com/MHidayatz/Streamlit_Diabetes_Prediction/main/Data/diabetes_data_upload.csv?token=GHSAT0AAAAAABYFMA5FKOW5GE225XC5D2SMYZQRFHA")
    df_encoded = load_data(r"https://raw.githubusercontent.com/MHidayatz/Streamlit_Diabetes_Prediction/main/Data/diabetes_data_upload_clean.csv?token=GHSAT0AAAAAABYFMA5F76R3ZOGHWDB4UC3CYZQRGNA")
    freq_df = load_data(r"https://raw.githubusercontent.com/MHidayatz/Streamlit_Diabetes_Prediction/main/Data/freqdist_of_age_data.csv?token=GHSAT0AAAAAABYFMA5F4XZLHQDMAQDJHI22YZQRG2A")

    submenu = st.sidebar.selectbox("Submenu", ["Descriptive","Plots"])
    if submenu == "Descriptive":
        st.dataframe(df)

        with st.expander("Data Types"):
            dtype = explore(df)
            st.write(dtype)

        with st.expander("Descriptive Summary"):
            st.dataframe(df_encoded.describe())
            
        with st.expander("Gender Distribution"):
            st.dataframe(df['Gender'].value_counts())

        with st.expander("Class Distribution"):
            st.dataframe(df['class'].value_counts())

    elif submenu == "Plots":
        st.subheader("Plots")

        # Layout
        col1, col2 = st.columns([2,1])

        with col1:
            # Gender Distribution
            with st.expander("Dist Plot of Gender"):
                # Seaborn
                fig = plt.figure()
                sns.countplot(data = df, x='Gender')
                st.pyplot(fig)
                
                gen_df = df['Gender'].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ['Gender Type','Counts']
                st.dataframe(gen_df)
                p01 = px.pie(gen_df,names='Gender Type',values='Counts')
                st.plotly_chart(p01,use_container_width=True)

            # For Class Distribution
            with st.expander("Dist Plot of Class"):                
                # Seaborn
                fig = plt.figure()
                sns.countplot(data = df, x='class')
                st.pyplot(fig)

        
        with col2:
            with st.expander("Gender Distribution"):
                st.dataframe(gen_df)            
            
            with st.expander("Class Distribution"):
                st.dataframe(df['class'].value_counts())
        
        # Freq Dist
        with st.expander("Frequency Dist of Age"):
            st.dataframe(freq_df)
            p2 =px.bar(freq_df, x='Age',y='count')
            st.plotly_chart(p2)

        # Outlier Detection
        with st.expander("Outlier Detection Plot"):
            fig = plt.figure()
            sns.boxplot(x = df['Age'])
            st.pyplot(fig)

            p3 = px.box(df,y='Age', color="Gender")
            st.plotly_chart(p3)

        # CorRelation
        with st.expander("Correlation Plot"):
            corr_matrix = df_encoded.corr()
            fig = plt.figure(figsize=(20,10))
            sns.heatmap(corr_matrix,annot=True)
            st.pyplot(fig)

            p4 = px.imshow(corr_matrix)
            st.plotly_chart(p4)