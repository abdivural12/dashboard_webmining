import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import folium_static

st.title('Visualisation des réductions de prix par région')

# Charger les données
@st.cache_data
def load_data():
    file_path = 'C:/Users/Abdi/Desktop/web_min_juin_24_projet/total_out_clean.csv'  # Assurez-vous que le fichier est dans le même répertoire que ce script
    data = pd.read_csv(file_path)
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

# Normaliser les données de réduction de prix
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
