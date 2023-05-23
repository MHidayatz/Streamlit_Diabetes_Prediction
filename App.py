#Core pkg
import streamlit as st
import streamlit.components.v1 as stc
st.set_page_config(page_title="ML Web App", page_icon=":tada:", layout="wide")
from streamlit_option_menu import option_menu

# Eda Pkg
import pandas as pd

# Utils
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

# Import Mini app
from eda_App import run_eda_app
from ml_App import run_ml_app

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Early Stage DM Risk Data App </h1>
		<h4 style="color:white;text-align:center;">Diabetes </h4>
		</div>
		"""

# Use local CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("Style/style.css")

def main():
    stc.html(html_temp)
    menu = ["Home", "EDA", "ML", "About"]
    choice = option_menu(
        menu_title = None,
        options = menu,
        icons = ["house", "book", "robot", "envelope"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal"
    )

    if choice =="Home":
        st.write("""
			### Early Stage Diabetes Risk Predictor App
			This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
			#### Datasource
				- https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
			#### App Content
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App

			""")
    elif choice =="EDA":
        run_eda_app()
    elif choice =="ML":
        run_ml_app()
    else:
        st.subheader("About")
        st.header(":mailbox: Get In Touch With Me!")
        
        contact_form = """
        <form action="https://formsubmit.co/m.hidayatz86@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your Email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
        </form>
        """

        st.markdown(contact_form, unsafe_allow_html=True)
        st.write("Created with [Form Submit](https://formsubmit.co/)")
        
if __name__ == '__main__':
	main()