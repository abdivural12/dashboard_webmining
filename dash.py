import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import folium_static
from PIL import Image

st.title('Visualisation du clustering des réductions de prix par région')

# Ajouter les logos
st.image("https://moodle.msengineering.ch/pluginfile.php/1/core_admin/logo/0x150/1643104191/logo-mse.png", width=250)
st.image("https://www.hes-so.ch/typo3conf/ext/wng_site/Resources/Public/HES-SO/img/logo_hesso_master_tablet.svg", width=250)

# Texte d'introduction
st.write("Cette interface a été créée dans le but de visualiser les résultats du clustering des réductions de prix et les résultats des modèles de prédiction.")

# Charger les données
@st.cache_data
def load_data():
    data = pd.read_csv('total_out_clean.csv')
    data['price_reduction'] = data['old_price'] - data['price']
    return data

data = load_data()

# Obtenir les coordonnées pour chaque région
geolocator = Nominatim(user_agent="geoapiExercises")

@st.cache_data
def get_coordinates(region):
    try:
        location = geolocator.geocode(region)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.write(f"Erreur pour la région {region}: {e}")
    return None, None

unique_regions = data['region'].unique()
coordinates = {region: get_coordinates(region) for region in unique_regions}

data['latitude'] = data['region'].map(lambda region: coordinates[region][0])
data['longitude'] = data['region'].map(lambda region: coordinates[region][1])

# Filtrer les lignes avec des coordonnées valides
data_with_coords = data.dropna(subset=['latitude', 'longitude'])

# Vérifier si les données avec coordonnées ne sont pas vides
if data_with_coords.empty:
    st.error("Aucune donnée valide après le filtrage des coordonnées. Veuillez vérifier les données d'entrée.")
else:
    # Normaliser les données de réduction de prix
    if not data_with_coords[['price_reduction']].empty:
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_with_coords[['price_reduction']])
        
        # Appliquer le clustering K-Means
        kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
        clusters = kmeans.fit_predict(data_normalized)

        # Ajouter les informations de cluster au DataFrame
        data_with_coords['cluster'] = clusters

        # Créer une carte centrée sur l'Espagne
        map = folium.Map(location=[40.4168, -3.7038], zoom_start=5)

        # Ajouter des points pour chaque région
        for idx, row in data_with_coords.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"{row['region']}: {row['price_reduction']}",
                color=['red', 'blue', 'green'][row['cluster']],
                fill=True,
                fill_color=['red', 'blue', 'green'][row['cluster']]
            ).add_to(map)

        # Afficher la carte
        st.write("Carte des réductions de prix par région")
        folium_static(map)
    else:
        st.error("Les données de réduction de prix sont vides après la normalisation.")

# Charger les images
image1_path = "ExampleSet.jpeg"
image2_path = "ExampleSet2.jpeg"

image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

# Afficher les images côte à côte
st.write("Graphiques:")
col1, col2 = st.columns(2)

with col1:
    st.image(image1, caption='Graphique 1: Modèle de Régression Linéaire', use_column_width=True)

with col2:
    st.image(image2, caption='Graphique 2: Modèle de Régression par Forêt Aléatoire', use_column_width=True)
