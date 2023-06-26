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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



#1. Import
data = pd.read_csv('data/books_limpio_def.csv',index_col=[0])
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

        fig = px.line(books_per_year, x='Año', y='Cantidad')
        fig.update_xaxes(title='Año')
        fig.update_yaxes(title='Cantidad de libros')

        st.plotly_chart(fig)

    # Puntuacion promedio por género
    def create_average_rating_by_genre_chart():
        datos_subplot_1 = data.groupby('genero_1')['average_rating'].mean().reset_index()
        fig = go.Figure(data=[go.Bar(x=datos_subplot_1['average_rating'], y=datos_subplot_1['genero_1'], marker=dict(color=color_palette), orientation='h')])
        fig.update_layout(
            xaxis_title="Puntuación promedio"
        )
        st.plotly_chart(fig)

    # Paginas promedio por género
    def create_average_pages_by_genre_chart():
        datos_subplot_2 = data.groupby('genero_1')['pages'].mean().reset_index()
        fig = go.Figure(data=[go.Bar(x=datos_subplot_2['pages'], y=datos_subplot_2['genero_1'], marker=dict(color=color_palette), orientation='h')])

        fig.update_layout(
            xaxis_title="Promedio de páginas"
        )
        st.plotly_chart(fig)

    
    # Libros mejor puntuados
    def create_top_rated_books_chart(genre):
        filtered_data = data[(data['genero_1'] == genre)]
        average_ratings = filtered_data.groupby('title')['average_rating'].mean().reset_index()
        top_10_books = average_ratings.nlargest(10, 'average_rating')
        fig = px.bar(top_10_books, x='average_rating', y='title', orientation='h',
                    text=top_10_books['average_rating'].round(2),
                    labels={'average_rating': 'Calificación promedio'},
                    title=f'<b>Top 10 libros con mejor puntuación en el Género: {genre}</b>',
                    color='title', color_discrete_sequence=color_palette)

        fig.update_layout(showlegend=False)
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig)


    # top 10 autores por género 
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
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig)

    if __name__ == '__main__':
        st.header('Libros por año')
        create_books_per_year_chart()
        st.header('Páginas promedio por género')
        create_average_pages_by_genre_chart()
        st.header('Puntuación promedio por género')
        create_average_rating_by_genre_chart()
        st.header('Mejor puntuados')
        genres = data['genero_1'].unique()
        selected_genre = st.selectbox('Selecciona un género', genres)
        create_top_rated_books_chart(selected_genre)
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
        st.write('De la concatenación de esas tres tablas se obtuvieron más de 100.000 registros. A partir del título y del código ISBN de Goodreads se cruzó con el dataset original y logramos obtener la descripción, las páginas y el género de 7.000 libros, de los 9.000 que teníamos orginalmente.')
        
        st.write('Con la combinación de las tablas hecha, continuamos haciendo la limpieza de los datos.')
        st.write('En primer lugar, trabajamos para rellenar los valores nulos y en los registros que no pudimos decidimos quitarlos del dataset.')
        st.write('Luego, encontramos descripciones de libros en diferentes idiomas, decidimos traducir todas al inglés utilizando la librería de Python googletrans.')
        st.write('A continuación, notamos que la columna género en realidad contenía una lista de géneros a los que pertenecía el libro. Para facilitar el análisis decidimos ordenar las listas de géneros poniendo en primer lugar el género más repetido de todo el dataset y en último el menos.')
        st.write('Con el género ordenado, tomamos los 4 primeros y los dividimos en 4 columnas diferentes, así nos quedamos con los 4 géneros "más importantes" de cada libro.')
        st.write('El siguiente paso fue trabajar sobre la columna descripción para quitar palabras y frases repetidas en muchos registros que no aportaran al análisis, así cómo una limpieza de stopwords y puntuación, la lematización y normalización de los strings.')
        st.write('Por último, se quitaron libros duplicados donde se repitiese el título y el código ISBN.')

        st.header('2. Modelado')
        st.write('Una vez finalizada la preparación de los datos comenzamos con el armado de los sistemas de recomendación.')
        
        # Recomendacion por colaboracion
        st.subheader('Sistema de Recomendación usando Modelos de Matrix Factorization')
        st.write('Estos modelos parten de una Matriz de Puntuación de los Usuarios, donde las filas son los Ítems, las columnas son los Usuarios y los valores representan la puntuación que cada Usuario le asignó a cada Ítem.')
        st.write('Luego, se busca una pareja de matrices cuyo producto dé como resultado la Matriz de Puntuación de los Usuarios.')
        st.write('Estas matrices se componen de la siguiente manera:')
        st.markdown('- Las filas son los Ítems y las columnas son los Features.')
        st.markdown('- Las filas son los Features y las columnas son los Usuarios.')
        st.write('Habiendo encontrado estas matrices, el Modelo es capaz de predecir cuál será la puntuación que cada Usuario le asignaría a cada Ítem multiplicándolas.')
        st.write('Para saber qué Modelo de Matrix Factorization utilizar, calculamos la dispersión de la matriz, que está dada por el porcentaje de puntuaciones que le faltan a dicha matriz. En este caso, el porcentaje fue del 99,82%, ')
        st.write('La dispersión puede traducirse en una mala performance del modelo. Por lo tanto, es necesario elegir un modelo que pueda manejar la escasez de interacciones entre usuarios e ítems. Es por esto que se eligió el modelo ALS.')
        st.write('Otras ventajas de este modelo son:')
        st.markdown('- Puede incorporar restricciones y regularizaciones para evitar el sobreajuste.')
        st.markdown('- Es poco sensible a outliers.')
        st.markdown('- Es más rápido que los métodos SVD y SGD.')
        st.markdown('- Es escalable a grandes conjuntos de datos.')
        st.write('La siguiente decisión a tomar para ejecutar el Modelo es la cantidad de Features (k) que se requerirán. Elegimos un k = 190, porque las matrices muy dispersas requieren un k grande. Sin embargo, sabemos que esto implica un riesgo de sobreajuste y una performance pobre.')

        # Recomendacion por contenido



    if __name__ == '__main__':
        model_backstage()


#####################################################################################################################################


# Pagina 4 = Modelo
elif selected == 'Encontrá tu libro':
    st.title('Encontrá tu próximo libro')
    st.header("Modelo basado en colaboración")
    user_number = st.text_input("Número de usuario", "")

    st.header("Modelo basado en contenido")
    book_titles = data['title'].unique()
    selected_book_title = st.text_input('Ingresa un título de libro', value='', key='book_title_input')

    def find_similar_books(book_title, num_similar_books=3):
        data.reset_index(drop=True, inplace=True)
        book_index = data.loc[data['title'] == book_title].index[0]
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(data['texto_lemmatizado'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        book_similarities = cosine_sim[book_index]
        similar_books_indices = book_similarities.argsort()[::-1][1:num_similar_books+1]
        similar_books = data.loc[similar_books_indices, ['title', 'genero_1', 'genero_2', 'pages', 'average_rating']]
        return similar_books

    # Verificar si el título ingresado existe en el dataset
    if selected_book_title.strip() not in book_titles:
        st.warning('Por favor, ingresa un título válido.')
    else:
        if __name__ == '__main__':
            similar_books = find_similar_books(selected_book_title, num_similar_books=3)
            st.write('**Libros similares:**')
            for i, book in similar_books.iterrows():
                st.write(f'**{book.title}**')
                st.markdown(f'- Género: {book.genero_1} - {book.genero_2} ')
                st.markdown(f'- Páginas: {book.pages}')
                st.markdown(f'- Rating: {book.average_rating}')

