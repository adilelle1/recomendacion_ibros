import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from plotly.subplots import make_subplots
import plotly.graph_objects as go




#1. Import
df = pd.read_csv('book_data_procesado.csv',index_col=[0])

#2. Titulo de pagina
st.set_page_config(page_title="Sistema de recomendación de libros")

#3. Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Home', 'Data visualization', 'Armado del modelo', 'Tu libro'],
    )


#####################################################################################################################################


# Pagina 1 = Home
if selected == 'Home':
    st.title('Sistema de recomendación de libros')
    st.write('Encontramos tu próximo libro favorito.')
    st.image('book.jpg')

    st.header('Problemática y objetivos')
    st.write('Ante la abrumadora cantidad de información que se puede encontrar hoy en día en los medios digitales, puede sentirse algo complicado encontrar qué libro leer.')
    st.write('Por eso, buscamos desarrollar un sistema que ayude a los amantes de la literatura.')
    
    st.header('Dataset')
    st.write('El conjunto de datos utilizado, es un subset de datos de la base de la página web [Goodreads](https://www.goodreads.com/).')

    st.write('Para este proyecto se utiliza una base de 9.000 libros y 981.756 puntajes que realizaron usuarios acerca de los mismos.')
    st.write("A continuación podemos ver cómo se compone el set de datos")
    st.dataframe(df.head())

    st.subheader("\n Descripcion de columnas.")
    st.markdown("\n **id** :  Id de la tabla.")
    st.markdown("\n **book_id** :  Id del libro, se utiliza para conectar con tabla de ratings.")
    st.markdown("\n **best_book_id** :  ?")
    st.markdown("\n **work_id** :  ?")
    st.markdown("\n **books_count** :  Cantidad de libros. ?")
    st.markdown("\n **isbn** :  Id de Goodreads.")
    st.markdown("\n **isbn13** :  Id de Goodreads.")
    st.markdown("\n **authors** :  Nombre de los autores del libro.")
    st.markdown("\n **original_publication_year** :  Año original de publicación.")
    st.markdown("\n **original_title** :  Título original del libro.")
    st.markdown("\n **title** :  Título del libro.")
    st.markdown("\n **language_code** :  Código de idioma en el que fue escrito el libro.")
    st.markdown("\n **average_rating** :  Puntuación promedio en Goodreads.")
    st.markdown("\n **ratings_count** :  Cantidad de puntuaciones en Goodreads.")
    st.markdown("\n **work_ratings_count** :  ?")
    st.markdown("\n **work_text_reviews_count** :  ?")
    st.markdown("\n **ratings_1** :  Cantidad de puntuaciones de 1 sobre 5.")
    st.markdown("\n **ratings_2** :  Cantidad de puntuaciones de 2 sobre 5.")
    st.markdown("\n **ratings_3** :  Cantidad de puntuaciones de 3 sobre 5.")
    st.markdown("\n **ratings_4** :  Cantidad de puntuaciones de 4 sobre 5.")
    st.markdown("\n **ratings_5** :  Cantidad de puntuaciones de 5 sobre 5.")
    st.markdown("\n **description** :  Descripción del libro.")
    st.markdown("\n **genre** :  Género/s del libro.")
    st.markdown("\n **pages** :  Cantidad de páginas del libro.")



#####################################################################################################################################


# Pagina 2 = Graficos
elif selected == 'Data visualization':
    st.title('Data visualization')

    #histplot
    col_histplot = st.sidebar.selectbox('Columna - Histplot',['country','location_type', 'age_of_respondent','household_size','relationship_with_head','marital_status','education_level','job_type'])
    def graf_hist():
        fig = px.histogram(df, x= col_histplot, color=col_histplot, color_discrete_sequence=px.colors.qualitative.Set2).update_xaxes(categoryorder='total descending')
        fig.update_layout(
            autosize=False,
            width=900,
            height=600,
            bargap= 0.2,
            yaxis = dict(tickfont = dict(size=18)),
            xaxis = dict(tickfont = dict(size=18)),
            showlegend=False,
            )
        st.plotly_chart(fig)
        
        st.write('---')
        st.header('Conclusiones del análisis')
        st.markdown('Al realizar el análisis de distribuciones y correlación, fue posible observar que:')
        st.write('---')

        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)


#####################################################################################################################################


# Pagina 3 = Comparación de modelos
elif selected == 'Armado del modelo':
    def model_backstage():
        st.title('Construyendo un modelo de recomendación')
        st.write('En esta sección repasaremos el trabajo realizado sobre los datos y los pasos que se realizaron para construir el modelo.')

        st.header('1. Preprocesamiento')
        st.write('Fue necesario realizar un proceso detallado de limpieza de los datos, ya que el dataset original contenía valores faltantes o incorrectos en columnas importantes para el armado del modelo.')
        st.write('El objetivo principal de esta etapa fue la mejora de las columnas descripción, páginas y género.')
        st.write('La columna descripción originalmente no se encontraba en el dataset, fue necesario hacer una combinación con otras fuentes para obtenerla. Se utilizaron 3 tablas adicionales.')
        st.write('Para algunos registros fue necesario realizar una búsqueda de la descripción, ya que a pesar del join de tablas no se encontró ningún valor.')
        st.write('Para las columnas de páginas y géneros el proceso fue similar.')
        ###### Agregar lo que hizo meli de traducción

        
        st.header('2. Modelo a elegir')

    if __name__ == '__main__':
        model_backstage()


#####################################################################################################################################


# Pagina 4 = Modelo
elif selected == 'Tu libro':
    st.title('Encontrá tu próximo libro')
    def inputs():
        st.sidebar.header('Model inputs')
        country = st.sidebar.selectbox('Country', df.country.unique())	
        location_type = st.sidebar.selectbox('Location type', df.location_type.unique())
        cellphone_access = st.sidebar.selectbox('Cellphone access', df.cellphone_access.unique())
        household_size = st.sidebar.number_input('Household size', 1) 
        age_of_respondent = st.sidebar.number_input('Age', 16) 
        gender_of_respondent = st.sidebar.selectbox('Gender', df.gender_of_respondent.unique())
        relationship_with_head = st.sidebar.selectbox('Relationship with head', df.relationship_with_head.unique())
        marital_status = st.sidebar.selectbox('Marital status', df.marital_status.unique()) 
        education_level = st.sidebar.selectbox('Education level', df.education_level.unique())
        job_type = st.sidebar.selectbox('Job type', df.job_type.unique()) 
        button = st.sidebar.button('Try model!')
        return country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type, button

    def get_data(country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type):
            data_inputs = {'country': country, 
                    'location_type': location_type, 
                    'cellphone_access': cellphone_access, 
                    'household_size': household_size, 
                    'age_of_respondent':age_of_respondent, 
                    'gender_of_respondent': gender_of_respondent, 
                    'relationship_with_head': relationship_with_head, 
                    'marital_status': marital_status, 
                    'education_level': education_level, 
                    'job_type': job_type}
            data= pd.DataFrame(data_inputs, index=[0])
            return data

    def print_results():
        country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type, button = inputs()
        if button:
            st.header('Probando el modelo con los datos ingresados')
            st.write('') 
            trial_data = get_data(country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type)
            
            # imprimimos el df con los inputs
            st.write('Estos son los datos que ingresaste:')
            st.dataframe(trial_data)

            with open('financial_inclusion.pkl', 'rb') as clf_inclusion:
                modelo_inclusion = pickle.load(clf_inclusion)
            # Prediccion usando el trial data con lo insertado en el form
            if modelo_inclusion.predict(trial_data) == 1:
                st.write('---')
                st.markdown('<h4 style="text-align: center; color: Green">El individuo se encuentra bancarizado</h4>',unsafe_allow_html=True)
                st.write('---')
            else:
                st.write('---')
                st.markdown('<h4 style="text-align: center; color: Red">El individuo no se encuentra bancarizado</h4>', unsafe_allow_html=True, )
                st.write('---')

    if __name__ == '__main__':
        print_results()  