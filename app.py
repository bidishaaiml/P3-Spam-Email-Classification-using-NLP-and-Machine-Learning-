import streamlit as st
import pickle

model=pickle.load(open('spam12345.pkl','rb'))
cv=pickle.load(open('vec12345.pkl','rb'))

def main():
    st.title("Email Spam Classifier Application")
    st.write("This is a Machine Learning applicaton to classify email as sapm")
    st.subheader("Classification")
    user_input=st.text_area("Enter an email to classify",height=100)
    if st.button("Classify"):
        if user_input:
            data=[user_input]
            print(data)
            vec=cv.transform(data).toarray()
            result=model.predict(vec)
            if result[0]==0:
                st.success("This is Not a Spam Email")
            else:
                st.error("This is a Spam Email")
    else:
        st.write("Please enter an email to classify")
main()
