# docker run -p 8000:8000 -it python bash 
# pip install --upgrade pip
# apt-get install cmake
# pip install streamlit # error
# pip install ollama
import streamlit as st
from ollama import chat

st.title("Простой Ollama чат")

# Поле для ввода запроса пользователем
user_input = st.text_input("Введите ваш запрос:")

if user_input:
    # Отправляем запрос в модель через Ollama
    response = chat(
        model="llama3.1",  # или другую модель, которую ты установил
        messages=[{"role": "user", "content": user_input}]
    )

    # Отображаем ответ модели
    st.write("Ответ модели:")
    st.write(response['message']['content'])
