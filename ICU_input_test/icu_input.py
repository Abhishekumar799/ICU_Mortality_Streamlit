import streamlit as st
import pandas as pd
import numpy as np

def num_add(num1,num2):
    num = num1+num2
    return num
def ICU_test():
    st.title('ICU model Input testing')

    user_input = st.text_input("Enter your name:")
    num1 = st.number_input("Enter your num1:", min_value=0, max_value=120, step=1, value=30)
    num2 = st.number_input("Enter your num2:", min_value=0, max_value=120, step=1, value=30)
    color = st.selectbox("Select your favorite color:", ["Red", "Green", "Blue"])
    selected_date = st.date_input("Select a date:")
    st.button("Submit")


    st.write(f"name is {user_input} and age is {num_add(num1,num2)} and color is{color} ")
ICU_test()
#print(a)
# def ICU_Input(a,b):
#     c = a+b
#     return c*5


# a = int(input("enter a number"))
# b = int(input("enter b number"))


# #print(c)

# d = ICU_Input(a,b)
# print("number d is", d)