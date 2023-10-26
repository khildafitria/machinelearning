import pickle
import streamlit as st

bodyfat = pickle.load(open('estimasi_Lemak_Tubuh.sav', 'rb'))

st.title('Perhitungan Lemak Tubuh')

Density = st.number_input('Kepadatan')
Age = st.number_input('Usia')
Weight = st.number_input('Berat Badan (Dalam pound)')
Height = st.number_input('Tinggi Badan (Dalam INCH) ')
Neck = st.number_input('Ukuran Leher (Dalam cm)')
Chest = st.number_input('Ukuran Dada (Dalam cm)')
Abdomen = st.number_input('Ukuran Perut (Dalam cm)')
Hip = st.number_input('Ukuran Panggul (Dalam cm)')
Thigh = st.number_input('Ukuran Paha (Dalam cm)')
Knee = st.number_input('Ukuran Lingkar Lutut (Dalam cm)')
Ankle = st.number_input('Ukuran Lingkar Pergelangan Kaki (Dalam cm)')
Biceps = st.number_input('Ukuran Lingkar Lengan Atas (Dalam cm)')
Foream = st.number_input('Ukuran Lingkar Lengan Bawah (Dalam cm)')
Wrist = st.number_input('Ukuran Lingkar Pergelangan Tangan (Dalam cm)')

body_fat = ''

if st.button('Hitung'):
    bodyfatcount = bodyfat.predict([[Density, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Foream, Wrist]])
    st.write('Perhitungan Lemak Tubuh : ', bodyfatcount)

