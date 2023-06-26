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
import plotly.colors as colors




#1. Import
data = pd.read_csv('data/books_limpio.csv',index_col=[0])
data_ratings = pd.read_csv('data/ratings.csv')

#2. Titulo de pagina
st.set_page_config(page_title="Sistema de recomendación de libros")

#3. Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Home', 'Visualizando los datos', 'Armado del modelo', 'Encontrá tu libro'],
    )


#####################################################################################################################################


# Pagina 1 = Home
if selected == 'Home':
    st.title('Sistema de recomendación de libros')
    st.write('Encontramos tu próximo libro favorito.')
    st.image('libro_sobre_cama.png', width=900)

    st.header('Problemática y objetivos')
    st.write('Ante la abrumadora cantidad de información que se puede encontrar hoy en día en los medios digitales, puede sentirse algo complicado encontrar qué libro leer.')
    st.write('Por eso, buscamos desarrollar un sistema que ayude a los amantes de la literatura.')
    st.write('En este proyecto trabajaremos desde el lugar de Goodreads, una reconocida plataforma donde los usuarios llevan registro y puntuan libros, donde creamos un nueva sistema de recomendación utilizando el historial del usuario.')
    
    st.header('Dataset')
    st.write('El conjunto de datos utilizado, es un subset de datos de la base de la página web [Goodreads](https://www.goodreads.com/).')

    st.write('Para este proyecto se utiliza una base de 6.000 libros y 981.756 puntuaciones que realizaron usuarios acerca de los mismos.')
    st.write("A continuación podemos ver cómo se componen los set de datos utilizados:")
    st.dataframe(data.head())
    st.dataframe(data_ratings.head())

    st.subheader("\n Descripcion de las columnas.")
    st.markdown("\n**Tabla Libros**")
    st.markdown("\n **id** :  Número de identificación de la tabla.")
    st.markdown("\n **book_id** :  Número de identificación del libro, se utiliza para conectar con tabla de ratings.")
    st.markdown("\n **books_count** :  Cantidad de libros. ?")
    st.markdown("\n **isbn** :  Número de identificación de Goodreads.")
    st.markdown("\n **isbn13** :  Número de identificación de Goodreads.")
    st.markdown("\n **authors** :  Nombre de los autores del libro.")
    st.markdown("\n **original_publication_year** :  Año original de publicación.")
    st.markdown("\n **original_title** :  Título original del libro.")
    st.markdown("\n **title** :  Título del libro.")
    st.markdown("\n **language_code** :  Código de idioma en el que fue escrito el libro.")
    st.markdown("\n **average_rating** :  Puntuación promedio en Goodreads.")
    st.markdown("\n **ratings_count** :  Cantidad de puntuaciones en Goodreads.")
    st.markdown("\n **ratings_1** :  Cantidad de puntuaciones de 1 sobre 5.")
    st.markdown("\n **ratings_2** :  Cantidad de puntuaciones de 2 sobre 5.")
    st.markdown("\n **ratings_3** :  Cantidad de puntuaciones de 3 sobre 5.")
    st.markdown("\n **ratings_4** :  Cantidad de puntuaciones de 4 sobre 5.")
    st.markdown("\n **ratings_5** :  Cantidad de puntuaciones de 5 sobre 5.")
    st.markdown("\n **description** :  Descripción del libro.")
    st.markdown("\n **pages** :  Cantidad de páginas del libro.")
    st.markdown("\n **genre** :  Género/s del libro.")
    st.markdown("\n **genre_ordenado** :  Géneros del libro ordenados por género más repetido en el dataset.")
    st.markdown("\n **genre_1** :  Primer género del libro (por importancia en las repeticiones en el dataset)")
    st.markdown("\n **genre_2** :  Segundo género del libro (por importancia en las repeticiones en el dataset).")
    st.markdown("\n **genre_3** :  Tercer género del libro (por importancia en las repeticiones en el dataset).")
    st.markdown("\n **genre_4** :  Cuarto género del libro (por importancia en las repeticiones en el dataset).")

    st.markdown("\n\n**Tabla Ratings**")
    st.markdown("\n **book_id** :  Número de identificación del libro.")
    st.markdown("\n **user_id** :  Número de identificación del usuario.")
    st.markdown("\n **rating** :  Puntuación del libro del 1 al 5.")

#####################################################################################################################################


# Pagina 2 = Graficos
elif selected == 'Visualizando los datos':
    st.title('Visualizando los datos')

    color_palette = colors.qualitative.Light24

    # libros por anio
    def create_books_per_year_chart():
        books_per_year = data.loc[data['original_publication_year'] > 1800, 'original_publication_year'].value_counts().sort_index().reset_index()
        books_per_year.columns = ['Año', 'Cantidad']

        fig = px.line(books_per_year, x='Año', y='Cantidad', title='<b>Libros por año</b>')
        fig.update_xaxes(title='Año')
        fig.update_yaxes(title='Cantidad de libros')

        st.plotly_chart(fig)
    
    # Libros mejor puntuados
    def create_top_rated_books_chart():
        top_10_most_rated_books = data.sort_values('ratings_count', ascending=False).head(10)
        fig = go.Figure(data=[go.Bar(
            x=top_10_most_rated_books['ratings_count'],
            y=top_10_most_rated_books['title'],
            orientation='h',
            marker=dict(color=color_palette))])

        fig.update_layout(
            title="<b>Top 10 libros con más puntuaciones</b>",
            xaxis_title="Número de Valoraciones",
            barmode='stack',
            yaxis_categoryorder='total ascending')
        
        st.plotly_chart(fig)

    # Puntuacion promedio por género
    def create_average_rating_by_genre_chart():
        datos_subplot_1 = data.groupby('genero_1')['average_rating'].mean().reset_index()
        fig = go.Figure(data=[go.Bar(x=datos_subplot_1['average_rating'], y=datos_subplot_1['genero_1'], marker=dict(color=color_palette), orientation='h')])
        fig.update_layout(
            title="<b>Puntuación promedio por género</b>",
            xaxis_title="Puntuación promedio"
        )
        st.plotly_chart(fig)

    # Paginas promedio por género
    def create_average_pages_by_genre_chart():
        datos_subplot_2 = data.groupby('genero_1')['pages'].mean().reset_index()
        fig = go.Figure(data=[go.Bar(x=datos_subplot_2['pages'], y=datos_subplot_2['genero_1'], marker=dict(color=color_palette), orientation='h')])

        fig.update_layout(
            title="<b>Número de páginas promedio por género</b>",
            xaxis_title="Promedio de páginas"
        )
        st.plotly_chart(fig)

    # top 10 autores por género 
    genres = data['genero_1'].unique()
    def puntuacion_autores_por_genero(genre):
        filtered_data = data[(data['genero_1'] == genre) & (~data['authors'].str.contains(','))]
        average_ratings = filtered_data.groupby('authors')['average_rating'].mean().reset_index()
        top_20_authors = average_ratings.nlargest(10, 'average_rating')
        fig = px.bar(top_20_authors, x='average_rating', y='authors', orientation='h',
                    text=top_20_authors['average_rating'].round(2),
                    labels={'average_rating': 'Calificación promedio'},
                    title=f'<b>Top 10 Autores con mejor puntuación en el Género: {genre}</b>',
                    color='authors', color_discrete_sequence=color_palette)

        fig.update_layout(showlegend=False)
        fig.update_xaxes(automargin=True, title=None)
        st.plotly_chart(fig)

    if __name__ == '__main__':
        st.header('Libros por año')
        create_books_per_year_chart()
        st.header('Mejor puntuados')
        create_top_rated_books_chart()
        st.header('Páginas promedio por género')
        create_average_pages_by_genre_chart()
        st.header('Puntuación promedio por género')
        create_average_rating_by_genre_chart()
        st.header('Top 10 autores')
        selected_genre = st.selectbox('Selecciona un género', genres)
        puntuacion_autores_por_genero(selected_genre)

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
elif selected == 'Econtrá tu libro':
    st.title('Encontrá tu próximo libro')
    st.sidebar.title("Ingrese su número de usuario")
    user_number = st.sidebar.text_input("Número de usuario", "")
    st.sidebar.write("Número de usuario ingresado:", user_number)

    
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