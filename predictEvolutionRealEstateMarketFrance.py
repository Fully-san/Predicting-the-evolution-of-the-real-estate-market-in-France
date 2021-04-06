import altair as alt
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import re
import seaborn as sns
import streamlit as st
import time
from scipy.stats import linregress
from scipy.stats import pearsonr

@st.cache(show_spinner=False)
def load_csv(filename, usecols=None, dtype=None, sep=','):
    return pd.read_csv(filename, usecols=usecols, dtype=dtype, sep=sep)

def createCityDataframe(dataset, listCodeDepartment, listCommune, listTypeLocal):
    df = dataset.loc[
            (dataset['Code departement'].isin(listCodeDepartment))
        &   (dataset['Commune'].isin(listCommune))
        &   (dataset['Nature mutation'].isin(['Vente', "Vente en l'état futur d'achèvement"]))
        &   (dataset['Type local'].isin(listTypeLocal))
        &   (dataset['Nombre de lots'] >= 1)]

    df['Surface Carrez du 1er lot'] = df['Surface Carrez du 1er lot'].fillna(0)
    df['Surface Carrez du 2eme lot'] = df['Surface Carrez du 2eme lot'].fillna(0)

    df.dropna(subset=['Valeur fonciere'], inplace=True)

    for i in range(len(df)):
        if df.iloc[i]["Surface Carrez du 1er lot"] == 0:
            df.iloc[i, df.columns.get_loc("Surface Carrez du 1er lot")] = df.iloc[i]["Surface Carrez du 2eme lot"]

    df.rename(columns={"Date mutation": "date", "Nature mutation": "typeTransaction", "Valeur fonciere": "price", "Surface Carrez du 1er lot": "area", "Type local":"typeLocal", "Nombre pieces principales":"numberRooms", "Surface terrain":"landArea"}, inplace=True)

    df["price"] = pd.to_numeric(df["price"].astype(str).str[:-3], errors='coerce')
    df["numberRooms"] = pd.to_numeric(df["numberRooms"].astype(str).str[:-2], errors='coerce')
    df["area"] = pd.to_numeric(df["area"].str.replace(',', '.'), errors='coerce')

    df["date"] = df["date"].str.split('/').str[2] + '-' + df["date"].str.split('/').str[1]

    df.drop(columns=['Surface Carrez du 2eme lot'], inplace=True)
    df.dropna(subset=['area'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.insert(3, 'pricePerM²', round(df['price'] / df['area'], 2))

    return df

def removeOutliers(df):
    q_prix_low = df["price"].quantile(0.01)
    q_prix_hi  = df["price"].quantile(0.99)
    q_surface_low = df["area"].quantile(0.01)
    q_surface_hi  = df["area"].quantile(0.99)
    q_nombrePieces_low = df["numberRooms"].quantile(0.01)
    q_nombrePieces_hi  = df["numberRooms"].quantile(0.99)

    df = df[
            (df["price"] < q_prix_hi)
        &   (df["price"] > q_prix_low)
        &   (df["area"] < q_surface_hi)
        &   (df["area"] > q_surface_low)
        &   (df["numberRooms"] < q_nombrePieces_hi)
        &   (df["numberRooms"] > q_nombrePieces_low)]

    return df

def createCityEWMA12PriceDataframe(df, years):
    data = []

    for year in years:
        for i in range(1,13):
            date = str(year) + '-' + str(i).zfill(2)
            ordinalDate = pd.to_datetime(date).toordinal()
            mean = round(df.loc[df['date'] == date, 'pricePerM²'].mean(), 2)
            date = pd.to_datetime(date)
            data.append([date, ordinalDate, mean])

    # Create the pandas DataFrame 
    new_df = pd.DataFrame(data, columns = ['date', 'ordinalDate', 'Average Monthly Price'])
    new_df.dropna(subset=['Average Monthly Price'], inplace=True)
    new_df['EWMA-12 Average Monthly Price'] = new_df['Average Monthly Price'].ewm(span=12).mean()

    return new_df

def drawTop15(data):
    top15_df = data

    st.write("")
    st.write("Evolution of the price per m² of the 15 largest cities in France between 2016 and 2019 (Without Strasbourg which is in Bas-Rhin).")

    sortCity = top15_df.loc[top15_df['date'] == '2016-01-01', ['price', 'city']]
    sortCity.sort_values(by=['price'], inplace=True, ascending=False)
    sortCity = sortCity['city'].to_numpy()

    selection = alt.selection_multi(fields=['city'], bind='legend')

    base = alt.Chart(top15_df, title="City TOP 15 (2016-2019)").encode(
        x=alt.X('date:T',
            axis=alt.Axis(title=None)
        ),
        y=alt.Y('price:Q',
            scale=alt.Scale(zero=False),
            axis=alt.Axis(title='price per m²')
        ),
        color=alt.Color('city', scale=alt.Scale(scheme='yellowgreenblue', reverse=True), sort=sortCity, legend=alt.Legend(title="")),
        tooltip=['city']
    )

    points = base.mark_circle().encode(
        opacity=alt.value(0)
    )
    
    lines = base.mark_line(interpolate='basis').encode(
        size=alt.value(3),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
    ).add_selection(
        selection
    ).properties(
        height=500
    )

    c = alt.layer(
        lines, points
    )

    st.altair_chart(c, use_container_width=True)

st.set_page_config(layout="wide")
st.title('Evolution of the real estate market in France')

st.write("Source of data: [Dataset] (https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/).")
st.write("The data cover the entire territory of metropolitan France, except for the departments of Bas-Rhin (67), Haut-Rhin (68) and Moselle (57).")

st.sidebar.title('Options')

typeLocal = st.sidebar.multiselect('Select the type of property you want:', ['Apartment', 'House', 'Outbuilding', 'Commercial premises'], default=['Apartment', 'House'], key='typeLocal')
cities = st.sidebar.text_area("Enter the different cities by going to the line:", value='', key="listCity")

codes = [i for i in range(1, 96) if i not in [55, 67, 68] ]
listCodeDepartment = st.sidebar.multiselect('Select the code of the department where your cities are located', codes)

st.sidebar.text('Loading of datasets (quite slow):')
progressBar = st.sidebar.progress(0)

submit = st.sidebar.button('Submit')

drawTop15(pd.read_csv('data/top15.csv'))

columnList = ['Code departement', 'Commune', 'Date mutation', 'Nature mutation', 'Nombre de lots', 'Nombre pieces principales', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Type local', 'Valeur fonciere']
dataframeList = []
fileList = ['data/valeursfoncieres-2016.txt', 'data/valeursfoncieres-2017.txt', 'data/valeursfoncieres-2018.txt', 'data/valeursfoncieres-2019.txt']
urlList = ["https://www.data.gouv.fr/fr/datasets/r/0ab442c5-57d1-4139-92c2-19672336401c", "https://www.data.gouv.fr/fr/datasets/r/7161c9f2-3d91-4caf-afa2-cfe535807f04", "https://www.data.gouv.fr/fr/datasets/r/1be77ca5-dc1b-4e50-af2b-0240147e0346", "https://www.data.gouv.fr/fr/datasets/r/3004168d-bec4-44d9-a781-ef16f41856a2", "https://www.data.gouv.fr/fr/datasets/r/90a98de0-f562-4328-aa16-fe0dd1dca60f"] 

for i in range(len(fileList)):
    if os.path.isfile(fileList[i]):
        dataframeList.append(load_csv(fileList[i], columnList, sep='|'))
    else:
        dataframeList.append(load_csv(urlList[i], columnList, sep='|'))

    progressBar.progress(i / len(fileList))

progressBar.progress(100)

if submit:
    listTypeLocal = []
    if 'Apartment' in typeLocal:
        listTypeLocal.append('Appartement')
    if 'House' in typeLocal:
        listTypeLocal.append('Maison')
    if 'Outbuilding' in typeLocal:
        listTypeLocal.append('Dépendance')
    if 'Commercial premises' in typeLocal:
        listTypeLocal.append('Local industriel. commercial ou assimilé')

    listCity = [x.upper() for x in cities.split('\n')]
    if 'PARIS' in listCity:
        parisList = ['PARIS 01','PARIS 02','PARIS 03','PARIS 04','PARIS 05','PARIS 06','PARIS 07','PARIS 08','PARIS 09','PARIS 10','PARIS 11','PARIS 12','PARIS 13','PARIS 14','PARIS 15','PARIS 16','PARIS 17','PARIS 18','PARIS 19','PARIS 20']
        listCity.extend(parisList)
        listCity.remove('PARIS')
    elif 'LYON' in listCity:
        lyonList = ['LYON 1ER', 'LYON 2EME', 'LYON 3EME', 'LYON 4EME', 'LYON 5EME', 'LYON 6EME', 'LYON 7EME', 'LYON 8EME', 'LYON 9EME']
        listCity.extend(lyonList)
        listCity.remove('LYON')
    elif 'MARSEILLE' in listCity:
        marseilleList = ['MARSEILLE 1ER', 'MARSEILLE 2EME', 'MARSEILLE 3EME', 'MARSEILLE 4EME', 'MARSEILLE 5EME', 'MARSEILLE 6EME', 'MARSEILLE 7EME', 'MARSEILLE 8EME', 'MARSEILLE 9EME', 'MARSEILLE 10EME', 'MARSEILLE 11EME', 'MARSEILLE 12EME', 'MARSEILLE 13EME', 'MARSEILLE 14EME', 'MARSEILLE 15EME', 'MARSEILLE 16EME']
        listCity.extend(marseilleList)
        listCity.remove('MARSEILLE')

    cleanDataframeList = []
    for dataframe in dataframeList:
        cleanDataframeList.append(removeOutliers(createCityDataframe(dataframe, listCodeDepartment, listCity, listTypeLocal)))

    df = pd.concat(cleanDataframeList).sort_values(by='date', ascending=True)

    if df.empty:
        st.title("There are no results for your search.")
    else:
        source = createCityEWMA12PriceDataframe(df.sort_values(by='date', ascending=True), ['2016','2017','2018', '2019'])

        a, b, r, p_value, std_err = linregress(source['ordinalDate'], source['EWMA-12 Average Monthly Price'])

        source['Linear Regression'] = a * source['ordinalDate'] + b
        source = source.set_index('date')
        source.drop(columns=['ordinalDate'], inplace=True)
        source = source.reset_index().melt('date', var_name='category', value_name='pricePerM²')

        title = "Evolution of the average monthly price per m² for the cities of"
        for city in listCity:
            title += ' ' + city.capitalize()

        line = alt.Chart(source, title=title).mark_line(interpolate='basis').encode(
            x=alt.X('date:T',
                axis=alt.Axis(title=None)
            ),
            y=alt.Y('pricePerM²:Q',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(title='price per m²')
            ), 
            color=alt.Color('category', scale=alt.Scale(scheme='yellowgreenblue'), legend=alt.Legend(title=""), sort=['Average Monthly Price', 'EWMA-12 Average Monthly Price', 'Linear Regression'])
        ).properties(
            height=400
        )

        c = alt.layer(
            line
        )

        st.altair_chart(c, use_container_width=True)

        text = ""
        for i in range(2015, 2031, 1):
            text += 'On the first January <span class="textColor">' + str(i) + '</span> the average price per m² for the cities of'
            for city in listCity:
                text += ' ' + city.capitalize()
            text += ' is estimated at '

            text += '<span class="textColor">' + str(round(a * dt.datetime(i, 1, 1).toordinal() + b)) + '</span> €.\n\n'

        st.markdown("""<style>.textColor {color: #45b4c2 !important;}</style>""", unsafe_allow_html=True)
        st.markdown(text, unsafe_allow_html=True)