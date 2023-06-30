import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder

def show_main_page():
    image = Image.open('image_cars.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Cars",
        page_icon=image,
    )

    st.write(
        """
        # Предсказание стоимости подержанных автомобилей
        Определяем стоимость подержанных автомобилей по их базовым характеристикам.
        """
    )

    st.image(image)

def process_side_bar_inputs():
    st.sidebar.header('Введите параметры автомобиля')
    user_input_df = sidebar_input_features()
    
    write_user_data(user_input_df)
    
    with open('model_cars.pkl', 'rb') as file:
        model = pickle.load(file)
    
    pred = model.predict(user_input_df)
    pred = round(np.expm1(pred)[0])
    
    st.write('## Предсказание')
    st.write(f'Стоимость вашего авто составляет:  {str(pred)}')


def sidebar_input_features():
    lst_make = ['AMBASSADOR', 'AUDI', 'BMW', 'CHEVROLET', 'DAEWOO', 'DATSUN', 'FIAT', 'FORCE', 'FORD', 'HONDA', 'HYUNDAI', 'ISUZU',
            'JAGUAR', 'JEEP', 'KIA', 'LAND ROVER','LEXUS', 'MAHINDRA', 'MARUTI', 'MERCEDES-BENZ', 'MG HECTOR', 'MITSUBISHI',
            'NISSAN', 'RENAULT', 'SKODA', 'TATA', 'TOYOTA', 'VOLKSWAGEN', 'VOLVO']
    with open('name_cars.txt') as file:
        name_cars = json.load(file)
        
    make = st.sidebar.selectbox("Марка автомобиля", lst_make)
    model_cars = st.sidebar.selectbox("Модель автомобиля", sorted(name_cars[make]))
    year = st.sidebar.slider("Год выпуска с завода-изготовителя", min_value=1994, max_value=2020, value=2015, step=1)
    km_driven = st.sidebar.number_input('Пробег на дату продажи, км', step=10.0)
    fuel = st.sidebar.selectbox('Тип топлива', ('Diesel', 'Petrol', 'CNG', 'LPG'))
    seller_type = st.sidebar.selectbox('Продавец', ('Dealer', 'Individual', 'Trustmark Dealer'))
    transmission = st.sidebar.selectbox('Тип трансмиссии', ('Automatic', 'Manual'))
    owner = st.sidebar.selectbox('Какой по счёту хозяин', ('First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'))
    mileage = st.sidebar.slider("Расход топлива, kmpl", min_value=0, max_value=35, value=19, step=1)
    engine = st.sidebar.slider('Рабочий объем двигателя', min_value=624, max_value=3604, value=1250, step=1)
    max_power = st.sidebar.slider('Пиковая мощность двигателя, bhp', min_value=32, max_value=300, value=80, step=1)
    torque1 = st.sidebar.slider('Крутящий момент, Nm', min_value=47, max_value=620, value=170, step=1)
    torque2 = st.sidebar.slider('Максимальные обороты, rpm', min_value=1300, max_value=5000, value=2400, step=50)
    seats = st.sidebar.slider('Количество посадочных мест', min_value=2, max_value=14, value=5, step=1)
    
    translatetion = {
        'Test Drive Car': 0,
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Automatic': 0,
        'Manual': 1
    }

    data = {
        'name': model_cars,
        'year': year,
        'km_driven': int(km_driven),
        'owner': translatetion[owner],
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats,
        'torque_1': torque1,
        'torque_2': torque2,
        'fuel_Diesel': 0,
        'fuel_LPG': 0,
        'fuel_Petrol': 0,
        'seller_type_Individual': 0,
        'seller_type_Trustmark Dealer': 0,
        'transmission_Manual': translatetion[transmission]
    }
    
    if fuel == 'Diesel':
        data['fuel_Diesel'] = 1
    elif fuel == 'Petrol':
        data['fuel_Petrol'] = 1
    elif fuel == 'LPG':
        data['fuel_LPG'] = 1
    
    if seller_type == 'Individual':
        data['seller_type_Individual'] = 1
    elif seller_type == 'Trustmark Dealer':
        data['seller_type_Trustmark Dealer'] = 1
    
    df = pd.DataFrame(data, index=[0])

    return df
    
def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)
    
    
show_main_page()
process_side_bar_inputs()
