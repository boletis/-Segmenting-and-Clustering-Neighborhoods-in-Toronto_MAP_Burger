
Segmenting and Clustering Neighborhoods in Toronto
-Create the above dataframe


Import Libraries


```python
import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
from pandas.io.json import json_normalize  # tranform JSON file into a pandas dataframe

import folium # map rendering library

# import k-means from clustering stage
from sklearn.cluster import KMeans

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
```

Import Table. Preprocess Data


```python
source = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M").text
soup = BeautifulSoup(source, 'lxml')

table = soup.find("table")
table_rows = table.tbody.find_all("tr")

res = []
for tr in table_rows:
    td = tr.find_all("td")
    row = [tr.text for tr in td]
    
    # Only process the cells that have an assigned borough. Ignore cells with a borough that is Not assigned.
    if row != [] and row[1] != "Not assigned":
        # If a cell has a borough but a "Not assigned" neighborhood, then the neighborhood will be the same as the borough.
        if "Not assigned" in row[2]: 
            row[2] = row[1]
        res.append(row)

# Dataframe with 3 columns
df = pd.DataFrame(res, columns = ["PostalCode", "Borough", "Neighborhood"])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A\n</td>
      <td>Not assigned\n</td>
      <td>Not assigned\n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A\n</td>
      <td>Not assigned\n</td>
      <td>Not assigned\n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A\n</td>
      <td>North York\n</td>
      <td>Parkwoods\n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A\n</td>
      <td>North York\n</td>
      <td>Victoria Village\n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A\n</td>
      <td>Downtown Toronto\n</td>
      <td>Regent Park, Harbourfront\n</td>
    </tr>
  </tbody>
</table>
</div>



Remove string "/n"


```python
df["PostalCode"] = df["PostalCode"].str.replace("\n","")
df["Borough"] = df["Borough"].str.replace("\n","")
df["Neighborhood"] = df["Neighborhood"].str.replace("\n","")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>




```python
df= df[~df['Borough'].isin(['Not assigned'])]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
    </tr>
  </tbody>
</table>
</div>



Group Boroughs by "Neighborhood" column


```python
df = df.groupby(["PostalCode", "Borough"])["Neighborhood"].apply(", ".join).reset_index()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Shape: ", df.shape)
```

    Shape:  (103, 3)


Create a coordinates Frame


```python
df_geo_coor = pd.read_csv("./Geospatial_Coordinates.csv")
df_geo_coor.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



  Merge the 2 dataframes "df" and "df_geo_coor" into one dataframe.


```python
df_toronto = pd.merge(df, df_geo_coor, how='left', left_on = 'PostalCode', right_on = 'Postal Code')

df_toronto.drop("Postal Code", axis=1, inplace=True)
df_toronto.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



Explore Toronto on Map


```python
address = "Toronto, ON"

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto city are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Toronto city are 43.6534817, -79.3839347.


Depict the map of Toronto using previous Data Frame


```python
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)
map_toronto

```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css%22/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_70c4f4710db24ea2b2130463850c26d1%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_70c4f4710db24ea2b2130463850c26d1%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_70c4f4710db24ea2b2130463850c26d1%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_70c4f4710db24ea2b2130463850c26d1%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B43.6534817%2C%20-79.3839347%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2010%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_2ee23d03629541a4aaaaa5f6109cf8ef%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



In order to further investigate the map, we will depict the markers


```python
for lat, lng, borough, neighborhood in zip(
        df_toronto['Latitude'], 
        df_toronto['Longitude'], 
        df_toronto['Borough'], 
        df_toronto['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  

map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css%22/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_70c4f4710db24ea2b2130463850c26d1%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_70c4f4710db24ea2b2130463850c26d1%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_70c4f4710db24ea2b2130463850c26d1%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_70c4f4710db24ea2b2130463850c26d1%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B43.6534817%2C%20-79.3839347%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2010%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_2ee23d03629541a4aaaaa5f6109cf8ef%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_eea748b97463463e8f34a720fac69544%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.806686299999996%2C%20-79.19435340000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_547c2426d6d841da94891270d5457b5c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d272c2694d4b448ab20910994eadd273%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d272c2694d4b448ab20910994eadd273%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMalvern%2C%20Rouge%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_547c2426d6d841da94891270d5457b5c.setContent%28html_d272c2694d4b448ab20910994eadd273%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_eea748b97463463e8f34a720fac69544.bindPopup%28popup_547c2426d6d841da94891270d5457b5c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bd2e0cc35f594949bf256e0c9c8df512%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7845351%2C%20-79.16049709999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a85a5b5f9f564cada602f215e35deec9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1d81f4f02ad84982a2aca822da8df494%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1d81f4f02ad84982a2aca822da8df494%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERouge%20Hill%2C%20Port%20Union%2C%20Highland%20Creek%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a85a5b5f9f564cada602f215e35deec9.setContent%28html_1d81f4f02ad84982a2aca822da8df494%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bd2e0cc35f594949bf256e0c9c8df512.bindPopup%28popup_a85a5b5f9f564cada602f215e35deec9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7c460b4ae8c54e5389844567a9ab454e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7635726%2C%20-79.1887115%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4b16a30921884b3ab44f0001ad6e9387%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ca7b7c9b38b34f6fb3a61a71c16608fb%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ca7b7c9b38b34f6fb3a61a71c16608fb%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGuildwood%2C%20Morningside%2C%20West%20Hill%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4b16a30921884b3ab44f0001ad6e9387.setContent%28html_ca7b7c9b38b34f6fb3a61a71c16608fb%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7c460b4ae8c54e5389844567a9ab454e.bindPopup%28popup_4b16a30921884b3ab44f0001ad6e9387%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c03b75a9bf594a1fbe5485690d928549%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7709921%2C%20-79.21691740000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_fb8226d9cebe4715aa96c58299fd5fd3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_63d6c23a7a4749dd81e25e5486e26bcf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_63d6c23a7a4749dd81e25e5486e26bcf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWoburn%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_fb8226d9cebe4715aa96c58299fd5fd3.setContent%28html_63d6c23a7a4749dd81e25e5486e26bcf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c03b75a9bf594a1fbe5485690d928549.bindPopup%28popup_fb8226d9cebe4715aa96c58299fd5fd3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_83f45ffd4a9148d697d37b72b20cce84%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.773136%2C%20-79.23947609999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_64876dc496014b869296b03f1034abd9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c69eb5c4221d4d9c92af0ddddef7256d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c69eb5c4221d4d9c92af0ddddef7256d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECedarbrae%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_64876dc496014b869296b03f1034abd9.setContent%28html_c69eb5c4221d4d9c92af0ddddef7256d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_83f45ffd4a9148d697d37b72b20cce84.bindPopup%28popup_64876dc496014b869296b03f1034abd9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7c339bddeae64f30a0a606cb4c23b23f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7447342%2C%20-79.23947609999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6d59d3e347d342c4859a49b5623ee1db%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_883a3ddffd984ee490b047dbc8f1fdde%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_883a3ddffd984ee490b047dbc8f1fdde%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EScarborough%20Village%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6d59d3e347d342c4859a49b5623ee1db.setContent%28html_883a3ddffd984ee490b047dbc8f1fdde%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7c339bddeae64f30a0a606cb4c23b23f.bindPopup%28popup_6d59d3e347d342c4859a49b5623ee1db%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2f967c01b0574cd48874f33400dbceb5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7279292%2C%20-79.26202940000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ca61a8c069684143850756caecabd33a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e696db035bed4fcca6df4c784d3cf276%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e696db035bed4fcca6df4c784d3cf276%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKennedy%20Park%2C%20Ionview%2C%20East%20Birchmount%20Park%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ca61a8c069684143850756caecabd33a.setContent%28html_e696db035bed4fcca6df4c784d3cf276%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2f967c01b0574cd48874f33400dbceb5.bindPopup%28popup_ca61a8c069684143850756caecabd33a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ef04f37ccd154e9a86a704363a5f21df%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.711111700000004%2C%20-79.2845772%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b06881dc26b146a08d92f6b88c44a8b1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d6718a6e51fa4130b55064ce1350f5ea%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d6718a6e51fa4130b55064ce1350f5ea%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGolden%20Mile%2C%20Clairlea%2C%20Oakridge%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b06881dc26b146a08d92f6b88c44a8b1.setContent%28html_d6718a6e51fa4130b55064ce1350f5ea%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ef04f37ccd154e9a86a704363a5f21df.bindPopup%28popup_b06881dc26b146a08d92f6b88c44a8b1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0fdbd571ce924b34b110d4493bbfb1c9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.716316%2C%20-79.23947609999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3b73e25932644316a1da0cee36865217%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b4ef7b2507a74bad82c77af2fce69c6e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b4ef7b2507a74bad82c77af2fce69c6e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECliffside%2C%20Cliffcrest%2C%20Scarborough%20Village%20West%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3b73e25932644316a1da0cee36865217.setContent%28html_b4ef7b2507a74bad82c77af2fce69c6e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0fdbd571ce924b34b110d4493bbfb1c9.bindPopup%28popup_3b73e25932644316a1da0cee36865217%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_804309f373234b23bfae295a855e2122%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.692657000000004%2C%20-79.2648481%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2147da5a6d2a4561bdf5a7b4286b1f43%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_38740bc04652474ea830339fc2390dd3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_38740bc04652474ea830339fc2390dd3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBirch%20Cliff%2C%20Cliffside%20West%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2147da5a6d2a4561bdf5a7b4286b1f43.setContent%28html_38740bc04652474ea830339fc2390dd3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_804309f373234b23bfae295a855e2122.bindPopup%28popup_2147da5a6d2a4561bdf5a7b4286b1f43%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_dfbcdee351ca4c09b4e66974403451d9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7574096%2C%20-79.27330400000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5673bf54df3846e58243f927993e0e2e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2e33e9eaf4bc4eeab36531333098a801%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2e33e9eaf4bc4eeab36531333098a801%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDorset%20Park%2C%20Wexford%20Heights%2C%20Scarborough%20Town%20Centre%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5673bf54df3846e58243f927993e0e2e.setContent%28html_2e33e9eaf4bc4eeab36531333098a801%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_dfbcdee351ca4c09b4e66974403451d9.bindPopup%28popup_5673bf54df3846e58243f927993e0e2e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e3d0703509504e3d8219b2537c8543a3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.750071500000004%2C%20-79.2958491%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_be44b07565834b0dadbc74552f388f1c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b01b1098ca1e461c9cb3d90fe111ee96%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b01b1098ca1e461c9cb3d90fe111ee96%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWexford%2C%20Maryvale%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_be44b07565834b0dadbc74552f388f1c.setContent%28html_b01b1098ca1e461c9cb3d90fe111ee96%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e3d0703509504e3d8219b2537c8543a3.bindPopup%28popup_be44b07565834b0dadbc74552f388f1c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_71af26ea69ed4a3295cd637d270e6012%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7942003%2C%20-79.26202940000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1747ea4811234ff4a01a892b4639a600%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bbb6507cd9794227b967995a30a60b3b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bbb6507cd9794227b967995a30a60b3b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAgincourt%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1747ea4811234ff4a01a892b4639a600.setContent%28html_bbb6507cd9794227b967995a30a60b3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_71af26ea69ed4a3295cd637d270e6012.bindPopup%28popup_1747ea4811234ff4a01a892b4639a600%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4554e0f90bfd4826b2360db01ac85fe3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7816375%2C%20-79.3043021%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1222c4fdc1794701bade5969f495756d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f53052692d78496d931254d33e783b9d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f53052692d78496d931254d33e783b9d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EClarks%20Corners%2C%20Tam%20O%26%2339%3BShanter%2C%20Sullivan%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1222c4fdc1794701bade5969f495756d.setContent%28html_f53052692d78496d931254d33e783b9d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4554e0f90bfd4826b2360db01ac85fe3.bindPopup%28popup_1222c4fdc1794701bade5969f495756d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b922a3bed0f6405e8742d6683ddeb658%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.8152522%2C%20-79.2845772%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4083aac77b87417cb390d1ab47936e05%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2c37c827c2374b11832646ca38eb6ee4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2c37c827c2374b11832646ca38eb6ee4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMilliken%2C%20Agincourt%20North%2C%20Steeles%20East%2C%20L%26%2339%3BAmoreaux%20East%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4083aac77b87417cb390d1ab47936e05.setContent%28html_2c37c827c2374b11832646ca38eb6ee4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b922a3bed0f6405e8742d6683ddeb658.bindPopup%28popup_4083aac77b87417cb390d1ab47936e05%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1466444ce1d94c588ef1d963b50274c9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.799525200000005%2C%20-79.3183887%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1f1df2fbec964de691e9d55d100b8f17%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dcf92f706881473abdb2774c6b153082%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dcf92f706881473abdb2774c6b153082%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESteeles%20West%2C%20L%26%2339%3BAmoreaux%20West%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1f1df2fbec964de691e9d55d100b8f17.setContent%28html_dcf92f706881473abdb2774c6b153082%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1466444ce1d94c588ef1d963b50274c9.bindPopup%28popup_1f1df2fbec964de691e9d55d100b8f17%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_37d1515ef8dd4ae095657db5d854e155%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.836124700000006%2C%20-79.20563609999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e63c98e956db42f69bd1168eb77b6bb6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1858949c5b8f45b4a3d2992abf0d8f7e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1858949c5b8f45b4a3d2992abf0d8f7e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUpper%20Rouge%2C%20Scarborough%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e63c98e956db42f69bd1168eb77b6bb6.setContent%28html_1858949c5b8f45b4a3d2992abf0d8f7e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_37d1515ef8dd4ae095657db5d854e155.bindPopup%28popup_e63c98e956db42f69bd1168eb77b6bb6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_83202e688d4347f1bcb32ad5e209ea69%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.8037622%2C%20-79.3634517%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c5ad6e44b8d94783b79a28e4e0962890%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0781bae59b2247b49dcaa81a0b2918e7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0781bae59b2247b49dcaa81a0b2918e7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHillcrest%20Village%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c5ad6e44b8d94783b79a28e4e0962890.setContent%28html_0781bae59b2247b49dcaa81a0b2918e7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_83202e688d4347f1bcb32ad5e209ea69.bindPopup%28popup_c5ad6e44b8d94783b79a28e4e0962890%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4ad4daedd5dd435eaf558616cb1ac035%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7785175%2C%20-79.3465557%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f5063963167a4e35bc1153acd3200d4e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_52f82098b2c14307853a5d144fb62e82%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_52f82098b2c14307853a5d144fb62e82%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFairview%2C%20Henry%20Farm%2C%20Oriole%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f5063963167a4e35bc1153acd3200d4e.setContent%28html_52f82098b2c14307853a5d144fb62e82%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4ad4daedd5dd435eaf558616cb1ac035.bindPopup%28popup_f5063963167a4e35bc1153acd3200d4e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a807f4b451034081b69a937c5ce4a5df%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7869473%2C%20-79.385975%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4b18d69b9ed841faa8cd8ebc9705eae4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b8bad25ea94741188d48970b77b44e56%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b8bad25ea94741188d48970b77b44e56%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBayview%20Village%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4b18d69b9ed841faa8cd8ebc9705eae4.setContent%28html_b8bad25ea94741188d48970b77b44e56%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a807f4b451034081b69a937c5ce4a5df.bindPopup%28popup_4b18d69b9ed841faa8cd8ebc9705eae4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4e624115089d4e9ca9aac1b72863306e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7574902%2C%20-79.37471409999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_eaaafb5341554a828a3dc7756fb58a43%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0e50eb0eaa80486297aba7da16fd777d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0e50eb0eaa80486297aba7da16fd777d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EYork%20Mills%2C%20Silver%20Hills%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_eaaafb5341554a828a3dc7756fb58a43.setContent%28html_0e50eb0eaa80486297aba7da16fd777d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4e624115089d4e9ca9aac1b72863306e.bindPopup%28popup_eaaafb5341554a828a3dc7756fb58a43%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b964190351464403bfb7ccc2a88056e4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.789053%2C%20-79.40849279999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_110ab481a6e34dbe9975340b44439dc2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c2c5aad298864fdf9dfffd7cc239e5b8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c2c5aad298864fdf9dfffd7cc239e5b8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWillowdale%2C%20Newtonbrook%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_110ab481a6e34dbe9975340b44439dc2.setContent%28html_c2c5aad298864fdf9dfffd7cc239e5b8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b964190351464403bfb7ccc2a88056e4.bindPopup%28popup_110ab481a6e34dbe9975340b44439dc2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a01ee0e94d874900896604892938e9a8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7701199%2C%20-79.40849279999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b2cec37782744ed1b452417612b263c1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fb22131904584ea180945dbc35e682bb%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_fb22131904584ea180945dbc35e682bb%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWillowdale%2C%20Willowdale%20East%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b2cec37782744ed1b452417612b263c1.setContent%28html_fb22131904584ea180945dbc35e682bb%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a01ee0e94d874900896604892938e9a8.bindPopup%28popup_b2cec37782744ed1b452417612b263c1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cf4beba5a3fa422f951387848bf36b3a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.752758299999996%2C%20-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_53f55dcbcdaa448591d8f5dd259be042%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3719f25587594c948d2dae8d9ad1a9c6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3719f25587594c948d2dae8d9ad1a9c6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EYork%20Mills%20West%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_53f55dcbcdaa448591d8f5dd259be042.setContent%28html_3719f25587594c948d2dae8d9ad1a9c6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_cf4beba5a3fa422f951387848bf36b3a.bindPopup%28popup_53f55dcbcdaa448591d8f5dd259be042%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_31e54baa291d4282a8fc5ad69da69db0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7827364%2C%20-79.4422593%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4be145a5eed04146a4b1150f0944f2f3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0e925a34f55b45d98ac005f6ad44e523%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0e925a34f55b45d98ac005f6ad44e523%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWillowdale%2C%20Willowdale%20West%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4be145a5eed04146a4b1150f0944f2f3.setContent%28html_0e925a34f55b45d98ac005f6ad44e523%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_31e54baa291d4282a8fc5ad69da69db0.bindPopup%28popup_4be145a5eed04146a4b1150f0944f2f3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_10bcfc4cb83c449f8ad1e424493bed85%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7532586%2C%20-79.3296565%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6095e01de7e34d69a1143ef7e6f40b63%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2a1af10448b8406e80f936b51fa54100%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2a1af10448b8406e80f936b51fa54100%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkwoods%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6095e01de7e34d69a1143ef7e6f40b63.setContent%28html_2a1af10448b8406e80f936b51fa54100%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_10bcfc4cb83c449f8ad1e424493bed85.bindPopup%28popup_6095e01de7e34d69a1143ef7e6f40b63%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_58fb6d72d0ea4d5fb78ecbb0e6a0a7ac%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.745905799999996%2C%20-79.352188%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ae9e3f2c96eb42ed8fdceda493d668b7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c045e0cc53e94529870c4a9088e2403e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c045e0cc53e94529870c4a9088e2403e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDon%20Mills%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ae9e3f2c96eb42ed8fdceda493d668b7.setContent%28html_c045e0cc53e94529870c4a9088e2403e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_58fb6d72d0ea4d5fb78ecbb0e6a0a7ac.bindPopup%28popup_ae9e3f2c96eb42ed8fdceda493d668b7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e23f2bedde3c49caaa1316dcb175c718%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.72589970000001%2C%20-79.340923%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_dc5f406857774b1483ff719f1cd58627%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1dcef709a8fc4906b8548e1a3b4094a9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1dcef709a8fc4906b8548e1a3b4094a9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDon%20Mills%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_dc5f406857774b1483ff719f1cd58627.setContent%28html_1dcef709a8fc4906b8548e1a3b4094a9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e23f2bedde3c49caaa1316dcb175c718.bindPopup%28popup_dc5f406857774b1483ff719f1cd58627%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_60857e33e8d842e2ad0fc17573431ebe%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7543283%2C%20-79.4422593%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_fb2df10207dc499595b33f6866ec4e46%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2e293bffb53145e890930acc9cfbf03a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2e293bffb53145e890930acc9cfbf03a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBathurst%20Manor%2C%20Wilson%20Heights%2C%20Downsview%20North%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_fb2df10207dc499595b33f6866ec4e46.setContent%28html_2e293bffb53145e890930acc9cfbf03a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_60857e33e8d842e2ad0fc17573431ebe.bindPopup%28popup_fb2df10207dc499595b33f6866ec4e46%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5370ad7506ae431db3a64864961b2a67%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7679803%2C%20-79.48726190000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b11052c9c8f24f1b80fa197d2540f0a5%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e1f8ea7857eb449e827be54896b58fcd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e1f8ea7857eb449e827be54896b58fcd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorthwood%20Park%2C%20York%20University%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b11052c9c8f24f1b80fa197d2540f0a5.setContent%28html_e1f8ea7857eb449e827be54896b58fcd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5370ad7506ae431db3a64864961b2a67.bindPopup%28popup_b11052c9c8f24f1b80fa197d2540f0a5%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_14d90d205ab74bd897fece74a23d573d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.737473200000004%2C%20-79.46476329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9c24c95a149a4c3f9cc3793dc73dcd95%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5957e4c7f89f43a8bd4219c474e0db9a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5957e4c7f89f43a8bd4219c474e0db9a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDownsview%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9c24c95a149a4c3f9cc3793dc73dcd95.setContent%28html_5957e4c7f89f43a8bd4219c474e0db9a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_14d90d205ab74bd897fece74a23d573d.bindPopup%28popup_9c24c95a149a4c3f9cc3793dc73dcd95%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_63832e81dba64b709a5ae1213e62f611%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7390146%2C%20-79.5069436%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c4f6312658b64334a0b3a4f44ad4b5f5%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_827ff73b9b9c4fc280e7b1a58c466053%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_827ff73b9b9c4fc280e7b1a58c466053%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDownsview%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c4f6312658b64334a0b3a4f44ad4b5f5.setContent%28html_827ff73b9b9c4fc280e7b1a58c466053%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_63832e81dba64b709a5ae1213e62f611.bindPopup%28popup_c4f6312658b64334a0b3a4f44ad4b5f5%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f6b29baf4cc245419e49f7b61b2d1bcb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7284964%2C%20-79.49569740000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6dca460164214fdf9a6ec8c1f3bb81fc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9c9955f62a9c41428177e99ef1b81b57%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9c9955f62a9c41428177e99ef1b81b57%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDownsview%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6dca460164214fdf9a6ec8c1f3bb81fc.setContent%28html_9c9955f62a9c41428177e99ef1b81b57%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f6b29baf4cc245419e49f7b61b2d1bcb.bindPopup%28popup_6dca460164214fdf9a6ec8c1f3bb81fc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_79471c52b9724eaa80f197cc1ef338d8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7616313%2C%20-79.52099940000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_eab5af3a0f6e456f822115f628e44b02%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0ca5933b29434507932fc8032c7e066b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0ca5933b29434507932fc8032c7e066b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDownsview%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_eab5af3a0f6e456f822115f628e44b02.setContent%28html_0ca5933b29434507932fc8032c7e066b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_79471c52b9724eaa80f197cc1ef338d8.bindPopup%28popup_eab5af3a0f6e456f822115f628e44b02%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d10c807d0fe14a66aff287eba7ba845e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.725882299999995%2C%20-79.31557159999998%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d3dfb9b4679646eaabbe3b7528e0234a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8b77c4e3d8c943bcac3b9fcc367c9aa0%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8b77c4e3d8c943bcac3b9fcc367c9aa0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EVictoria%20Village%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d3dfb9b4679646eaabbe3b7528e0234a.setContent%28html_8b77c4e3d8c943bcac3b9fcc367c9aa0%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d10c807d0fe14a66aff287eba7ba845e.bindPopup%28popup_d3dfb9b4679646eaabbe3b7528e0234a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_77404d5fb71d48979dc8ba8d4d06de4d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7063972%2C%20-79.309937%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c1e48da197e04b9e96400bd0fca44907%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f139ff775791444fb8862b17262187b8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f139ff775791444fb8862b17262187b8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkview%20Hill%2C%20Woodbine%20Gardens%2C%20East%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c1e48da197e04b9e96400bd0fca44907.setContent%28html_f139ff775791444fb8862b17262187b8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_77404d5fb71d48979dc8ba8d4d06de4d.bindPopup%28popup_c1e48da197e04b9e96400bd0fca44907%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_099b117c24594090abd37491dfd0afd5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.695343900000005%2C%20-79.3183887%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_aad8a481e16e4bfcbef44866a8d53b4c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_92b4dee892834b2982ef4eb23521f424%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_92b4dee892834b2982ef4eb23521f424%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWoodbine%20Heights%2C%20East%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_aad8a481e16e4bfcbef44866a8d53b4c.setContent%28html_92b4dee892834b2982ef4eb23521f424%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_099b117c24594090abd37491dfd0afd5.bindPopup%28popup_aad8a481e16e4bfcbef44866a8d53b4c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_46f8e99488f44581a48f53d6a135c37d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.67635739999999%2C%20-79.2930312%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_166e2cd182194ebe94eb8854e17a0fb6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1d67f6cbebd4486a85f4ffc63c940952%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1d67f6cbebd4486a85f4ffc63c940952%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Beaches%2C%20East%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_166e2cd182194ebe94eb8854e17a0fb6.setContent%28html_1d67f6cbebd4486a85f4ffc63c940952%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_46f8e99488f44581a48f53d6a135c37d.bindPopup%28popup_166e2cd182194ebe94eb8854e17a0fb6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9221f4f864784851af1348ee39111efd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7090604%2C%20-79.3634517%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4b2a052e37264abf9df181e92b841223%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_31cfe7d67ff3463790f80b08af4f2184%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_31cfe7d67ff3463790f80b08af4f2184%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELeaside%2C%20East%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4b2a052e37264abf9df181e92b841223.setContent%28html_31cfe7d67ff3463790f80b08af4f2184%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9221f4f864784851af1348ee39111efd.bindPopup%28popup_4b2a052e37264abf9df181e92b841223%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9376cf3bc6f14b9a887b98b9ce9f9f49%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7053689%2C%20-79.34937190000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f6ac7df1abde41f2bb79ee35d4065d99%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_05f8a8f0fe644ffa97b4ff5928e8e42b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_05f8a8f0fe644ffa97b4ff5928e8e42b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThorncliffe%20Park%2C%20East%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f6ac7df1abde41f2bb79ee35d4065d99.setContent%28html_05f8a8f0fe644ffa97b4ff5928e8e42b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9376cf3bc6f14b9a887b98b9ce9f9f49.bindPopup%28popup_f6ac7df1abde41f2bb79ee35d4065d99%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_97123c01172740b8b07c5a77535dacd5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.685347%2C%20-79.3381065%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c274c985b19147b190e83c3c6b181861%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b631ae105e5a40c89b59addbe6bc47bc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b631ae105e5a40c89b59addbe6bc47bc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Toronto%2C%20Broadview%20North%20%28Old%20East%20York%29%2C%20East%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c274c985b19147b190e83c3c6b181861.setContent%28html_b631ae105e5a40c89b59addbe6bc47bc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_97123c01172740b8b07c5a77535dacd5.bindPopup%28popup_c274c985b19147b190e83c3c6b181861%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_59b39a1c831d430f9a5b2b8227087356%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6795571%2C%20-79.352188%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e8fcc953d87f4112aed9abb6cc49cb52%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_72adb450a16546ed987d81e8ec9a904e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_72adb450a16546ed987d81e8ec9a904e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Danforth%20West%2C%20Riverdale%2C%20East%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e8fcc953d87f4112aed9abb6cc49cb52.setContent%28html_72adb450a16546ed987d81e8ec9a904e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_59b39a1c831d430f9a5b2b8227087356.bindPopup%28popup_e8fcc953d87f4112aed9abb6cc49cb52%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b43787bc268242ab874bfe8fd5b9227a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6689985%2C%20-79.31557159999998%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3f272da7f4cb4cbdbb4fb058d3d8d07c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1386939ccbb5482fb7547f1ac7fcaf05%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1386939ccbb5482fb7547f1ac7fcaf05%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EIndia%20Bazaar%2C%20The%20Beaches%20West%2C%20East%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3f272da7f4cb4cbdbb4fb058d3d8d07c.setContent%28html_1386939ccbb5482fb7547f1ac7fcaf05%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b43787bc268242ab874bfe8fd5b9227a.bindPopup%28popup_3f272da7f4cb4cbdbb4fb058d3d8d07c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_85f6d0c4a2fa4c17bbff518573747764%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6595255%2C%20-79.340923%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f3d321c52d6d44a1a85021e9122aa094%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_da6bbe481b39461eadc9aefc35a4332e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_da6bbe481b39461eadc9aefc35a4332e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStudio%20District%2C%20East%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f3d321c52d6d44a1a85021e9122aa094.setContent%28html_da6bbe481b39461eadc9aefc35a4332e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_85f6d0c4a2fa4c17bbff518573747764.bindPopup%28popup_f3d321c52d6d44a1a85021e9122aa094%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6695c4592cd3428883090ac08aaff2c5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7280205%2C%20-79.3887901%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3ca8b01805034663b818322f5824d3c9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b6e5493027ae4901a439067ac4f8e003%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b6e5493027ae4901a439067ac4f8e003%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELawrence%20Park%2C%20Central%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3ca8b01805034663b818322f5824d3c9.setContent%28html_b6e5493027ae4901a439067ac4f8e003%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6695c4592cd3428883090ac08aaff2c5.bindPopup%28popup_3ca8b01805034663b818322f5824d3c9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_48dc620b309c4589b65538cfc2faa10e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7127511%2C%20-79.3901975%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_183eceeda38d447ca9fa43c749b1be51%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8c78c6afd7114fcbaa47b7260d2da1ef%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8c78c6afd7114fcbaa47b7260d2da1ef%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%20North%2C%20Central%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_183eceeda38d447ca9fa43c749b1be51.setContent%28html_8c78c6afd7114fcbaa47b7260d2da1ef%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_48dc620b309c4589b65538cfc2faa10e.bindPopup%28popup_183eceeda38d447ca9fa43c749b1be51%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a051aa6314ef4d89b812802007d8850b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7153834%2C%20-79.40567840000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_006fa5b6455c4916983583b04048009f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_163762224a304358b5549f21af616b08%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_163762224a304358b5549f21af616b08%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Toronto%20West%2C%20%20Lawrence%20Park%2C%20Central%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_006fa5b6455c4916983583b04048009f.setContent%28html_163762224a304358b5549f21af616b08%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a051aa6314ef4d89b812802007d8850b.bindPopup%28popup_006fa5b6455c4916983583b04048009f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f2e85a7825f54356a6316757188b84eb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7043244%2C%20-79.3887901%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_bad01a86e31a4b3da894a9495f121d84%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ad2589c6d55345ff852ef10e647b1cad%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ad2589c6d55345ff852ef10e647b1cad%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%2C%20Central%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_bad01a86e31a4b3da894a9495f121d84.setContent%28html_ad2589c6d55345ff852ef10e647b1cad%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f2e85a7825f54356a6316757188b84eb.bindPopup%28popup_bad01a86e31a4b3da894a9495f121d84%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2be093c2612f4a5c85c783ad7b7ec096%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6895743%2C%20-79.38315990000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d688a675389f4fe891c9430a33d28fc1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f68443356297433790fad2cb191f4309%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f68443356297433790fad2cb191f4309%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMoore%20Park%2C%20Summerhill%20East%2C%20Central%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d688a675389f4fe891c9430a33d28fc1.setContent%28html_f68443356297433790fad2cb191f4309%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2be093c2612f4a5c85c783ad7b7ec096.bindPopup%28popup_d688a675389f4fe891c9430a33d28fc1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7cbb6c36dad3474da88b6e78339ebebf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.68641229999999%2C%20-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cd3bddce7e404c57860ce4166c528da5%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8c2e39fd07f949c081dbe26051c10def%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8c2e39fd07f949c081dbe26051c10def%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESummerhill%20West%2C%20Rathnelly%2C%20South%20Hill%2C%20Forest%20Hill%20SE%2C%20Deer%20Park%2C%20Central%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cd3bddce7e404c57860ce4166c528da5.setContent%28html_8c2e39fd07f949c081dbe26051c10def%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7cbb6c36dad3474da88b6e78339ebebf.bindPopup%28popup_cd3bddce7e404c57860ce4166c528da5%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d515800e648c4e6886ad21027ba066ca%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6795626%2C%20-79.37752940000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6cdd96a7b3ff443180c4b06b446c45bd%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c49088ffa073452e931f628142116069%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c49088ffa073452e931f628142116069%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERosedale%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6cdd96a7b3ff443180c4b06b446c45bd.setContent%28html_c49088ffa073452e931f628142116069%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d515800e648c4e6886ad21027ba066ca.bindPopup%28popup_6cdd96a7b3ff443180c4b06b446c45bd%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ffbcb3f84b624fa3b934d8b80ddf9238%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.667967%2C%20-79.3676753%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_39cca16130524c2191af55e0d30b0c1d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8d19bc4a063e42c09d8b7fc705146682%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8d19bc4a063e42c09d8b7fc705146682%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%2C%20Cabbagetown%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_39cca16130524c2191af55e0d30b0c1d.setContent%28html_8d19bc4a063e42c09d8b7fc705146682%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ffbcb3f84b624fa3b934d8b80ddf9238.bindPopup%28popup_39cca16130524c2191af55e0d30b0c1d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_97b82438d4d24db5a4e449bce376a22c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6658599%2C%20-79.38315990000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_bccb5a7b86da4939b9260e67d7c1a198%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8d9417c8b07b42d2bc4c90caa6b7b5e9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8d9417c8b07b42d2bc4c90caa6b7b5e9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChurch%20and%20Wellesley%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_bccb5a7b86da4939b9260e67d7c1a198.setContent%28html_8d9417c8b07b42d2bc4c90caa6b7b5e9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_97b82438d4d24db5a4e449bce376a22c.bindPopup%28popup_bccb5a7b86da4939b9260e67d7c1a198%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_48d6edf4fa8d4464a89fd2e471194ddf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6542599%2C%20-79.3606359%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_01faecfd91184327b9808b17ff3cdad5%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_265c783b7be946988bcd9e5bd5d0c7ad%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_265c783b7be946988bcd9e5bd5d0c7ad%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERegent%20Park%2C%20Harbourfront%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_01faecfd91184327b9808b17ff3cdad5.setContent%28html_265c783b7be946988bcd9e5bd5d0c7ad%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_48d6edf4fa8d4464a89fd2e471194ddf.bindPopup%28popup_01faecfd91184327b9808b17ff3cdad5%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_938d330746ab4fad870ea03ee673e9e6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6571618%2C%20-79.37893709999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e60e34e8d88543f3bb11efafbb4f6f1c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dc0bda3bb9854f6797b97f31d5db0f33%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dc0bda3bb9854f6797b97f31d5db0f33%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGarden%20District%2C%20Ryerson%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e60e34e8d88543f3bb11efafbb4f6f1c.setContent%28html_dc0bda3bb9854f6797b97f31d5db0f33%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_938d330746ab4fad870ea03ee673e9e6.bindPopup%28popup_e60e34e8d88543f3bb11efafbb4f6f1c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_639859ea2e0e470b804d7370fdcaa4f4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6514939%2C%20-79.3754179%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_53c75c3b3a8449a38552fb1a3a37fe39%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9f5760a9197e4f6c8e26c721824cbec1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9f5760a9197e4f6c8e26c721824cbec1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_53c75c3b3a8449a38552fb1a3a37fe39.setContent%28html_9f5760a9197e4f6c8e26c721824cbec1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_639859ea2e0e470b804d7370fdcaa4f4.bindPopup%28popup_53c75c3b3a8449a38552fb1a3a37fe39%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2d9be45519744de3bc64141f34e107d1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.644770799999996%2C%20-79.3733064%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2cb84db4d1c3430fb244fd50f2d595ff%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e560550bc2514226bb885313444b3f19%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e560550bc2514226bb885313444b3f19%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBerczy%20Park%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2cb84db4d1c3430fb244fd50f2d595ff.setContent%28html_e560550bc2514226bb885313444b3f19%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2d9be45519744de3bc64141f34e107d1.bindPopup%28popup_2cb84db4d1c3430fb244fd50f2d595ff%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6d6c6981d53a4006930fad0db8b74dd2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6579524%2C%20-79.3873826%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8011cf67a1784922ac55f2c4afb1c877%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d711eeacea704577882553d48585b0db%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d711eeacea704577882553d48585b0db%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Bay%20Street%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8011cf67a1784922ac55f2c4afb1c877.setContent%28html_d711eeacea704577882553d48585b0db%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6d6c6981d53a4006930fad0db8b74dd2.bindPopup%28popup_8011cf67a1784922ac55f2c4afb1c877%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e289e891445b4e7db84b9a4804263b61%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65057120000001%2C%20-79.3845675%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ad0bb0824072431385480dc31f46fee3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e6eb885ca08a40a2827e949a12670386%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e6eb885ca08a40a2827e949a12670386%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERichmond%2C%20Adelaide%2C%20King%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ad0bb0824072431385480dc31f46fee3.setContent%28html_e6eb885ca08a40a2827e949a12670386%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e289e891445b4e7db84b9a4804263b61.bindPopup%28popup_ad0bb0824072431385480dc31f46fee3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2d5264edf19b47fc84e53ef1764ab82e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6408157%2C%20-79.38175229999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e36ee907a6c64a43aa5ce0d875f3d69a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5de0340017e5428ebf60e509d2d87b07%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5de0340017e5428ebf60e509d2d87b07%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHarbourfront%20East%2C%20Union%20Station%2C%20Toronto%20Islands%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e36ee907a6c64a43aa5ce0d875f3d69a.setContent%28html_5de0340017e5428ebf60e509d2d87b07%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2d5264edf19b47fc84e53ef1764ab82e.bindPopup%28popup_e36ee907a6c64a43aa5ce0d875f3d69a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cb0d4d62b9ff4f3d9d13080856cf2903%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6471768%2C%20-79.38157640000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_922a983d46cd452fbf35677819a70911%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_85b4137b8b5f446598579c84a252cd49%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_85b4137b8b5f446598579c84a252cd49%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EToronto%20Dominion%20Centre%2C%20Design%20Exchange%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_922a983d46cd452fbf35677819a70911.setContent%28html_85b4137b8b5f446598579c84a252cd49%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_cb0d4d62b9ff4f3d9d13080856cf2903.bindPopup%28popup_922a983d46cd452fbf35677819a70911%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_17db5b7cd3664c30b6fd2b0733b1130c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6481985%2C%20-79.37981690000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2901600e79c848399d96b07264f51f23%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6409f783954d435b9ad617a75cf61496%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6409f783954d435b9ad617a75cf61496%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECommerce%20Court%2C%20Victoria%20Hotel%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2901600e79c848399d96b07264f51f23.setContent%28html_6409f783954d435b9ad617a75cf61496%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_17db5b7cd3664c30b6fd2b0733b1130c.bindPopup%28popup_2901600e79c848399d96b07264f51f23%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_40e4ba2273694fcca30fac36e0b6ac15%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7332825%2C%20-79.4197497%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5c97d406e0554b40af051b3db331e154%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_38232209d4a34a8ca4594bd39ae4002e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_38232209d4a34a8ca4594bd39ae4002e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBedford%20Park%2C%20Lawrence%20Manor%20East%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5c97d406e0554b40af051b3db331e154.setContent%28html_38232209d4a34a8ca4594bd39ae4002e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_40e4ba2273694fcca30fac36e0b6ac15.bindPopup%28popup_5c97d406e0554b40af051b3db331e154%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_83612d62a2af475aadd2907bcf6265d1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7116948%2C%20-79.41693559999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ca60eee1f767495cb2fae1815103a62d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_327e2b4c3d3540ce88e0eb2da3e9d46f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_327e2b4c3d3540ce88e0eb2da3e9d46f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERoselawn%2C%20Central%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ca60eee1f767495cb2fae1815103a62d.setContent%28html_327e2b4c3d3540ce88e0eb2da3e9d46f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_83612d62a2af475aadd2907bcf6265d1.bindPopup%28popup_ca60eee1f767495cb2fae1815103a62d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7ad8d63addda4c689516697afd470e7c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6969476%2C%20-79.41130720000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5e6b3cd81f354356a18acfb740807814%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_87ae0dd5c1c84595a3cc5278b62789b7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_87ae0dd5c1c84595a3cc5278b62789b7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EForest%20Hill%20North%20%26amp%3B%20West%2C%20Forest%20Hill%20Road%20Park%2C%20Central%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5e6b3cd81f354356a18acfb740807814.setContent%28html_87ae0dd5c1c84595a3cc5278b62789b7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7ad8d63addda4c689516697afd470e7c.bindPopup%28popup_5e6b3cd81f354356a18acfb740807814%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_035404c66f024e6da0e7a78f3f811b91%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6727097%2C%20-79.40567840000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_97aaafe7dde9468baff302e6d9eb6dae%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_781c484d56df46a1ad5f5eb16a395e4b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_781c484d56df46a1ad5f5eb16a395e4b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Annex%2C%20North%20Midtown%2C%20Yorkville%2C%20Central%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_97aaafe7dde9468baff302e6d9eb6dae.setContent%28html_781c484d56df46a1ad5f5eb16a395e4b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_035404c66f024e6da0e7a78f3f811b91.bindPopup%28popup_97aaafe7dde9468baff302e6d9eb6dae%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e2cc72cf0a494f999d1326a7a514712a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6626956%2C%20-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_12b2710320ca4aecac54481794650ea9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1e1a3a7eee574e95861285f9a49a82dc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1e1a3a7eee574e95861285f9a49a82dc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUniversity%20of%20Toronto%2C%20Harbord%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_12b2710320ca4aecac54481794650ea9.setContent%28html_1e1a3a7eee574e95861285f9a49a82dc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e2cc72cf0a494f999d1326a7a514712a.bindPopup%28popup_12b2710320ca4aecac54481794650ea9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b6c10ce6668e44aa834ad9ac3b79cca9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6532057%2C%20-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5dbc9283099d48d6865aae148e884501%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5262d5f96f49453c8110f53a2d0b1201%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5262d5f96f49453c8110f53a2d0b1201%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKensington%20Market%2C%20Chinatown%2C%20Grange%20Park%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5dbc9283099d48d6865aae148e884501.setContent%28html_5262d5f96f49453c8110f53a2d0b1201%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b6c10ce6668e44aa834ad9ac3b79cca9.bindPopup%28popup_5dbc9283099d48d6865aae148e884501%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_64672bb7dc404bada4c995fbb0649173%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6289467%2C%20-79.3944199%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5b57ab1f93de4932abd302480fe2fa72%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b92af41c275a48a2a92cc632f2778c48%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b92af41c275a48a2a92cc632f2778c48%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECN%20Tower%2C%20King%20and%20Spadina%2C%20Railway%20Lands%2C%20Harbourfront%20West%2C%20Bathurst%20Quay%2C%20South%20Niagara%2C%20Island%20airport%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5b57ab1f93de4932abd302480fe2fa72.setContent%28html_b92af41c275a48a2a92cc632f2778c48%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_64672bb7dc404bada4c995fbb0649173.bindPopup%28popup_5b57ab1f93de4932abd302480fe2fa72%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_91caf3637dc64bd293cc659e29b09b03%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6464352%2C%20-79.37484599999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_10f2c6a82c1b40a495cbff550aee5301%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_83cf83fc6c98407fa2eaf65be9ae14d3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_83cf83fc6c98407fa2eaf65be9ae14d3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStn%20A%20PO%20Boxes%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_10f2c6a82c1b40a495cbff550aee5301.setContent%28html_83cf83fc6c98407fa2eaf65be9ae14d3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_91caf3637dc64bd293cc659e29b09b03.bindPopup%28popup_10f2c6a82c1b40a495cbff550aee5301%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_eebbd9f5996e4676abc759b2d3d931a2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6484292%2C%20-79.3822802%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b630fa435bc440eeb61bf1e40a98aa6a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fd7eb41fdb154d08ba2034057e95c0f0%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_fd7eb41fdb154d08ba2034057e95c0f0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFirst%20Canadian%20Place%2C%20Underground%20city%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b630fa435bc440eeb61bf1e40a98aa6a.setContent%28html_fd7eb41fdb154d08ba2034057e95c0f0%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_eebbd9f5996e4676abc759b2d3d931a2.bindPopup%28popup_b630fa435bc440eeb61bf1e40a98aa6a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7c11a6fa97cd492eb84861509d710ef6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.718517999999996%2C%20-79.46476329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f323e56c74064fe6b53542674c3ff508%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8eb495b87f72424cbe83f10afc363e64%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8eb495b87f72424cbe83f10afc363e64%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELawrence%20Manor%2C%20Lawrence%20Heights%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f323e56c74064fe6b53542674c3ff508.setContent%28html_8eb495b87f72424cbe83f10afc363e64%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7c11a6fa97cd492eb84861509d710ef6.bindPopup%28popup_f323e56c74064fe6b53542674c3ff508%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b0a01f5473624d14ba6228ba2789526f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.709577%2C%20-79.44507259999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_16c50b8b085847b5850d36168e0136ea%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eb1ff687716e42ba993984223d8705a9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_eb1ff687716e42ba993984223d8705a9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGlencairn%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_16c50b8b085847b5850d36168e0136ea.setContent%28html_eb1ff687716e42ba993984223d8705a9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b0a01f5473624d14ba6228ba2789526f.bindPopup%28popup_16c50b8b085847b5850d36168e0136ea%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_37b319cc7dfc445e80fe91dd2aa34d5d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6937813%2C%20-79.42819140000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f1c45a7f399a4afc88473116a50a465d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e9948ef5dce34df5b30d8c1d1bb32847%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e9948ef5dce34df5b30d8c1d1bb32847%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHumewood-Cedarvale%2C%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f1c45a7f399a4afc88473116a50a465d.setContent%28html_e9948ef5dce34df5b30d8c1d1bb32847%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_37b319cc7dfc445e80fe91dd2aa34d5d.bindPopup%28popup_f1c45a7f399a4afc88473116a50a465d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3f62f1eeb1b4492bb1917429ef8304de%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6890256%2C%20-79.453512%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_06fd4360efa145dd9ebbe218bf0e1498%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_328bac8decaa420e8ef2d7a86b1535bb%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_328bac8decaa420e8ef2d7a86b1535bb%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECaledonia-Fairbanks%2C%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_06fd4360efa145dd9ebbe218bf0e1498.setContent%28html_328bac8decaa420e8ef2d7a86b1535bb%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3f62f1eeb1b4492bb1917429ef8304de.bindPopup%28popup_06fd4360efa145dd9ebbe218bf0e1498%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a3c8e4ef0f244444b732401c40502cda%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.669542%2C%20-79.4225637%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_176a4e5a5fc24692896c22d768150b59%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_24029dd1b9184e6d8f712f5c39de2249%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_24029dd1b9184e6d8f712f5c39de2249%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChristie%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_176a4e5a5fc24692896c22d768150b59.setContent%28html_24029dd1b9184e6d8f712f5c39de2249%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a3c8e4ef0f244444b732401c40502cda.bindPopup%28popup_176a4e5a5fc24692896c22d768150b59%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c3d7ebc6011a42f084ca7b19ba9a289b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.66900510000001%2C%20-79.4422593%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_11b33900479541b98038f449987500b3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_882a8adad8f346e8bf3acf3710d44117%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_882a8adad8f346e8bf3acf3710d44117%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDufferin%2C%20Dovercourt%20Village%2C%20West%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_11b33900479541b98038f449987500b3.setContent%28html_882a8adad8f346e8bf3acf3710d44117%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c3d7ebc6011a42f084ca7b19ba9a289b.bindPopup%28popup_11b33900479541b98038f449987500b3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e5ef7f9a66a5439687de8852d6013997%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.647926700000006%2C%20-79.4197497%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e59ce06b08a44413941dbe5dd8fc7cb2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_28023ea8233d434bbf723dfbe8bc8ea9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_28023ea8233d434bbf723dfbe8bc8ea9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELittle%20Portugal%2C%20Trinity%2C%20West%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e59ce06b08a44413941dbe5dd8fc7cb2.setContent%28html_28023ea8233d434bbf723dfbe8bc8ea9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e5ef7f9a66a5439687de8852d6013997.bindPopup%28popup_e59ce06b08a44413941dbe5dd8fc7cb2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3f4223f1bdf645f39d7044ad43f0e05f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6368472%2C%20-79.42819140000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6876f024e7994da7a14fff57fc2ec66e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f1f6079af67a43bc8dfd6a274bb095f2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f1f6079af67a43bc8dfd6a274bb095f2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBrockton%2C%20Parkdale%20Village%2C%20Exhibition%20Place%2C%20West%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6876f024e7994da7a14fff57fc2ec66e.setContent%28html_f1f6079af67a43bc8dfd6a274bb095f2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3f4223f1bdf645f39d7044ad43f0e05f.bindPopup%28popup_6876f024e7994da7a14fff57fc2ec66e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5ccd1ebbeaed47cf8f52e1ecc1694505%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.713756200000006%2C%20-79.4900738%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f7c930b524ab4bafa6217d986ad05f0a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_321697ce051b4ca89338dbb78cd837c3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_321697ce051b4ca89338dbb78cd837c3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Park%2C%20Maple%20Leaf%20Park%2C%20Upwood%20Park%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f7c930b524ab4bafa6217d986ad05f0a.setContent%28html_321697ce051b4ca89338dbb78cd837c3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5ccd1ebbeaed47cf8f52e1ecc1694505.bindPopup%28popup_f7c930b524ab4bafa6217d986ad05f0a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d5d5f3c4745b4bf0a045a4a427d1e838%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6911158%2C%20-79.47601329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_825e3391c77e40219fb1a282cc88db1a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dc55ac02c8ee4405b035290a56508f5c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dc55ac02c8ee4405b035290a56508f5c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDel%20Ray%2C%20Mount%20Dennis%2C%20Keelsdale%20and%20Silverthorn%2C%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_825e3391c77e40219fb1a282cc88db1a.setContent%28html_dc55ac02c8ee4405b035290a56508f5c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d5d5f3c4745b4bf0a045a4a427d1e838.bindPopup%28popup_825e3391c77e40219fb1a282cc88db1a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e006b0b649414123a97b95d6f95d04fd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.67318529999999%2C%20-79.48726190000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_13f9b6ab07fd4457a5144ee83a6544eb%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1481245b34ef4e57bad28f5c37b70d0d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1481245b34ef4e57bad28f5c37b70d0d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERunnymede%2C%20The%20Junction%20North%2C%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_13f9b6ab07fd4457a5144ee83a6544eb.setContent%28html_1481245b34ef4e57bad28f5c37b70d0d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e006b0b649414123a97b95d6f95d04fd.bindPopup%28popup_13f9b6ab07fd4457a5144ee83a6544eb%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2892b9cb40ce43e4bdd84c8c9f8b84d2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6616083%2C%20-79.46476329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_57d46d9bb2b849d0a7b4d11b2ccc78c4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4fa48d5234c64d6d8f94a011ac4f8eaf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4fa48d5234c64d6d8f94a011ac4f8eaf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHigh%20Park%2C%20The%20Junction%20South%2C%20West%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_57d46d9bb2b849d0a7b4d11b2ccc78c4.setContent%28html_4fa48d5234c64d6d8f94a011ac4f8eaf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2892b9cb40ce43e4bdd84c8c9f8b84d2.bindPopup%28popup_57d46d9bb2b849d0a7b4d11b2ccc78c4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ad27773f0d2245a4bd5c70dbdf738dc0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6489597%2C%20-79.456325%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1358856e8cfc4243b32518483808c051%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1bb36e579f8e4c73a26aa4fb1137f814%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1bb36e579f8e4c73a26aa4fb1137f814%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkdale%2C%20Roncesvalles%2C%20West%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1358856e8cfc4243b32518483808c051.setContent%28html_1bb36e579f8e4c73a26aa4fb1137f814%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ad27773f0d2245a4bd5c70dbdf738dc0.bindPopup%28popup_1358856e8cfc4243b32518483808c051%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1a87a167a549427a8b4d8abb551eccce%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6515706%2C%20-79.4844499%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f2c21b1410e9443e8442f0d59e56f361%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_10d018ea2738442c95c7c78003ed752f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_10d018ea2738442c95c7c78003ed752f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERunnymede%2C%20Swansea%2C%20West%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f2c21b1410e9443e8442f0d59e56f361.setContent%28html_10d018ea2738442c95c7c78003ed752f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1a87a167a549427a8b4d8abb551eccce.bindPopup%28popup_f2c21b1410e9443e8442f0d59e56f361%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_200565a90a904a8babc2151c55aa7c7f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6623015%2C%20-79.3894938%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a0f64e4c5d7340cb85ec3d01f43da8b6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2cb5a17a400649a5a808b863d29f5b3c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2cb5a17a400649a5a808b863d29f5b3c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EQueen%26%2339%3Bs%20Park%2C%20Ontario%20Provincial%20Government%2C%20Downtown%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a0f64e4c5d7340cb85ec3d01f43da8b6.setContent%28html_2cb5a17a400649a5a808b863d29f5b3c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_200565a90a904a8babc2151c55aa7c7f.bindPopup%28popup_a0f64e4c5d7340cb85ec3d01f43da8b6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fd4e6ea9252a405686a268f1825d8058%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6369656%2C%20-79.61581899999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_43dc3f62da1847a8a8531180a6587fb8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0c3cdc19adaf490f97b9a95000f6356e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0c3cdc19adaf490f97b9a95000f6356e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECanada%20Post%20Gateway%20Processing%20Centre%2C%20Mississauga%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_43dc3f62da1847a8a8531180a6587fb8.setContent%28html_0c3cdc19adaf490f97b9a95000f6356e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fd4e6ea9252a405686a268f1825d8058.bindPopup%28popup_43dc3f62da1847a8a8531180a6587fb8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c5ae56534e0a40e8ab7ab7a309322a01%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6627439%2C%20-79.321558%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1459721d998f4312a8aa3cca3ebdbe91%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_98a3ceb6d68f4766a4a745fc252f0003%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_98a3ceb6d68f4766a4a745fc252f0003%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBusiness%20reply%20mail%20Processing%20Centre%2C%20South%20Central%20Letter%20Processing%20Plant%20Toronto%2C%20East%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1459721d998f4312a8aa3cca3ebdbe91.setContent%28html_98a3ceb6d68f4766a4a745fc252f0003%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c5ae56534e0a40e8ab7ab7a309322a01.bindPopup%28popup_1459721d998f4312a8aa3cca3ebdbe91%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ef1d45b4d8bf49daad2d121a59624e04%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6056466%2C%20-79.50132070000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_056b13757362493d92a1cda5635a401d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_072530dd9d404bdb80c1dd6e8b0bee29%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_072530dd9d404bdb80c1dd6e8b0bee29%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENew%20Toronto%2C%20Mimico%20South%2C%20Humber%20Bay%20Shores%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_056b13757362493d92a1cda5635a401d.setContent%28html_072530dd9d404bdb80c1dd6e8b0bee29%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ef1d45b4d8bf49daad2d121a59624e04.bindPopup%28popup_056b13757362493d92a1cda5635a401d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2a99c836804e49e28ee237c5e543fdad%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.60241370000001%2C%20-79.54348409999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9fd45754521741479aa4a3de287b5e91%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ed11a879de154ca8b2d742ef69f78d3b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ed11a879de154ca8b2d742ef69f78d3b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAlderwood%2C%20Long%20Branch%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9fd45754521741479aa4a3de287b5e91.setContent%28html_ed11a879de154ca8b2d742ef69f78d3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2a99c836804e49e28ee237c5e543fdad.bindPopup%28popup_9fd45754521741479aa4a3de287b5e91%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_015b2938c63346bdaf4782399cb2e881%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.653653600000005%2C%20-79.5069436%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_58924b357d5a4002b4cce57be0ea3b7b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9c693d179a3d4f3280801ad43f8de791%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9c693d179a3d4f3280801ad43f8de791%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Kingsway%2C%20Montgomery%20Road%2C%20Old%20Mill%20North%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_58924b357d5a4002b4cce57be0ea3b7b.setContent%28html_9c693d179a3d4f3280801ad43f8de791%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_015b2938c63346bdaf4782399cb2e881.bindPopup%28popup_58924b357d5a4002b4cce57be0ea3b7b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bea0719a4cbe4054a58dbd66a5abf314%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6362579%2C%20-79.49850909999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6ca8581ed7e44b18bae545a387665b7a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_526a5002a9d748238d0198d4a48e7795%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_526a5002a9d748238d0198d4a48e7795%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EOld%20Mill%20South%2C%20King%26%2339%3Bs%20Mill%20Park%2C%20Sunnylea%2C%20Humber%20Bay%2C%20Mimico%20NE%2C%20The%20Queensway%20East%2C%20Royal%20York%20South%20East%2C%20Kingsway%20Park%20South%20East%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6ca8581ed7e44b18bae545a387665b7a.setContent%28html_526a5002a9d748238d0198d4a48e7795%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bea0719a4cbe4054a58dbd66a5abf314.bindPopup%28popup_6ca8581ed7e44b18bae545a387665b7a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b257195451ad477ca3a67e7b769cca91%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6288408%2C%20-79.52099940000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_89c21617083c436ebd13c84f3fecc1ea%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ead47c9727ff4fa6aefccd708bee737d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ead47c9727ff4fa6aefccd708bee737d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMimico%20NW%2C%20The%20Queensway%20West%2C%20South%20of%20Bloor%2C%20Kingsway%20Park%20South%20West%2C%20Royal%20York%20South%20West%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_89c21617083c436ebd13c84f3fecc1ea.setContent%28html_ead47c9727ff4fa6aefccd708bee737d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b257195451ad477ca3a67e7b769cca91.bindPopup%28popup_89c21617083c436ebd13c84f3fecc1ea%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fdaab4c9baf941bdada11bc442eac5b2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6678556%2C%20-79.53224240000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b7c673ae31bf49478d4cf1619d9f70d0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f976d705b806483ebebd466485b62e72%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f976d705b806483ebebd466485b62e72%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EIslington%20Avenue%2C%20Humber%20Valley%20Village%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b7c673ae31bf49478d4cf1619d9f70d0.setContent%28html_f976d705b806483ebebd466485b62e72%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fdaab4c9baf941bdada11bc442eac5b2.bindPopup%28popup_b7c673ae31bf49478d4cf1619d9f70d0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_15e02c9d8792455f9d130b0a75f839f3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6509432%2C%20-79.55472440000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_77b2cc87f6524b5c8f051139a2d72f54%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5faa926b738c4e1b910c6cd170134fc2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5faa926b738c4e1b910c6cd170134fc2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWest%20Deane%20Park%2C%20Princess%20Gardens%2C%20Martin%20Grove%2C%20Islington%2C%20Cloverdale%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_77b2cc87f6524b5c8f051139a2d72f54.setContent%28html_5faa926b738c4e1b910c6cd170134fc2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_15e02c9d8792455f9d130b0a75f839f3.bindPopup%28popup_77b2cc87f6524b5c8f051139a2d72f54%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cf07e0bffe194deba8ed50ace0be1e2a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6435152%2C%20-79.57720079999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5d91430f5a1844cf9763934e7bd9a20e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ac00ce63405c47c59a342d2a6aff93c9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ac00ce63405c47c59a342d2a6aff93c9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEringate%2C%20Bloordale%20Gardens%2C%20Old%20Burnhamthorpe%2C%20Markland%20Wood%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5d91430f5a1844cf9763934e7bd9a20e.setContent%28html_ac00ce63405c47c59a342d2a6aff93c9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_cf07e0bffe194deba8ed50ace0be1e2a.bindPopup%28popup_5d91430f5a1844cf9763934e7bd9a20e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d93a9353ebeb479bbcebc44be8c840be%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7563033%2C%20-79.56596329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6c8279f1ae4d486590a8c0098f28ae31%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b57fe1a6aa39406f8dccd38187cbd193%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b57fe1a6aa39406f8dccd38187cbd193%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHumber%20Summit%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6c8279f1ae4d486590a8c0098f28ae31.setContent%28html_b57fe1a6aa39406f8dccd38187cbd193%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d93a9353ebeb479bbcebc44be8c840be.bindPopup%28popup_6c8279f1ae4d486590a8c0098f28ae31%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_35a4a60a1a0945f5809eb8bd57a24af9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7247659%2C%20-79.53224240000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_820ee95c3bb041cf863fd8cb9c5f960d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9634212d7fcc4751afe9a76c43df33ce%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9634212d7fcc4751afe9a76c43df33ce%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHumberlea%2C%20Emery%2C%20North%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_820ee95c3bb041cf863fd8cb9c5f960d.setContent%28html_9634212d7fcc4751afe9a76c43df33ce%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_35a4a60a1a0945f5809eb8bd57a24af9.bindPopup%28popup_820ee95c3bb041cf863fd8cb9c5f960d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b06084a857864c85aa3ecdbf56a16b1c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.706876%2C%20-79.51818840000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5e1546862ba64fd68542bfafe18ed4ac%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8783b360b5834237ac34d353e3828648%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8783b360b5834237ac34d353e3828648%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWeston%2C%20York%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5e1546862ba64fd68542bfafe18ed4ac.setContent%28html_8783b360b5834237ac34d353e3828648%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b06084a857864c85aa3ecdbf56a16b1c.bindPopup%28popup_5e1546862ba64fd68542bfafe18ed4ac%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_59c5faf8afe5467a9eddcafe8d96f79b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.696319%2C%20-79.53224240000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9fe02e962c024f9eb748e3225642f749%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_479143d7f1144787b79670262a0476f3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_479143d7f1144787b79670262a0476f3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWestmount%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9fe02e962c024f9eb748e3225642f749.setContent%28html_479143d7f1144787b79670262a0476f3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_59c5faf8afe5467a9eddcafe8d96f79b.bindPopup%28popup_9fe02e962c024f9eb748e3225642f749%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9a5e268afee24fe593983c4688f53d66%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6889054%2C%20-79.55472440000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f297742ef4a54fa8b8e562dc2214c9ce%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_768800d076ea4d2688efa969150601dd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_768800d076ea4d2688efa969150601dd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKingsview%20Village%2C%20St.%20Phillips%2C%20Martin%20Grove%20Gardens%2C%20Richview%20Gardens%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f297742ef4a54fa8b8e562dc2214c9ce.setContent%28html_768800d076ea4d2688efa969150601dd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9a5e268afee24fe593983c4688f53d66.bindPopup%28popup_f297742ef4a54fa8b8e562dc2214c9ce%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a1844a2b8313440d88e7808d93fc51f5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.739416399999996%2C%20-79.5884369%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5e4a329cb8d84eb9a669db8b0d3f3415%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7cea8f7bce0043c49fd2561458624f46%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7cea8f7bce0043c49fd2561458624f46%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESouth%20Steeles%2C%20Silverstone%2C%20Humbergate%2C%20Jamestown%2C%20Mount%20Olive%2C%20Beaumond%20Heights%2C%20Thistletown%2C%20Albion%20Gardens%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5e4a329cb8d84eb9a669db8b0d3f3415.setContent%28html_7cea8f7bce0043c49fd2561458624f46%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a1844a2b8313440d88e7808d93fc51f5.bindPopup%28popup_5e4a329cb8d84eb9a669db8b0d3f3415%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_780d4d0fd87d45b9942bf5067dd081fc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.706748299999994%2C%20-79.5940544%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_70c4f4710db24ea2b2130463850c26d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3725da66d6c74155ace65a5f1cffb953%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_faf062da3f5a4bf59ed4636ab4be64ad%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_faf062da3f5a4bf59ed4636ab4be64ad%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorthwest%2C%20West%20Humber%20-%20Clairville%2C%20Etobicoke%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3725da66d6c74155ace65a5f1cffb953.setContent%28html_faf062da3f5a4bf59ed4636ab4be64ad%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_780d4d0fd87d45b9942bf5067dd081fc.bindPopup%28popup_3725da66d6c74155ace65a5f1cffb953%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
CLIENT_ID = 'KNFWX0XS0300Z0QS5J4AHBFNQXTVMLBJC0ULIV0TQQPAYOIW' # your Foursquare ID
CLIENT_SECRET = '5RDY2MP2HMXAWWCD3OZVTCKIW2S1YCK1HA3UCHF2I4DFQUPD' # your Foursquare Secret
ACCESS_TOKEN = '' # your FourSquare Access Token
VERSION = '20180604'
LIMIT = 30
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    Your credentails:
    CLIENT_ID: KNFWX0XS0300Z0QS5J4AHBFNQXTVMLBJC0ULIV0TQQPAYOIW
    CLIENT_SECRET:5RDY2MP2HMXAWWCD3OZVTCKIW2S1YCK1HA3UCHF2I4DFQUPD



```python
search_query = 'Burger'
radius = 5000
print(search_query + ' .... OK!')
```

    Burger .... OK!



```python
url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&oauth_token={}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude,ACCESS_TOKEN, VERSION, search_query, radius, LIMIT)
url
```




    'https://api.foursquare.com/v2/venues/search?client_id=KNFWX0XS0300Z0QS5J4AHBFNQXTVMLBJC0ULIV0TQQPAYOIW&client_secret=5RDY2MP2HMXAWWCD3OZVTCKIW2S1YCK1HA3UCHF2I4DFQUPD&ll=43.6534817,-79.3839347&oauth_token=&v=20180604&query=Burger&radius=5000&limit=30'



Send the GET Request and examine the results


```python
results = requests.get(url).json()
results
```




    {'meta': {'code': 200, 'requestId': '6017c2b7909c70607b860e1a'},
     'response': {'venues': [{'id': '5b7e6b38e179100039356f91',
        'name': 'Big Smoke Burger',
        'location': {'address': '260 Yonge St',
         'lat': 43.654907,
         'lng': -79.380873,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.654907,
           'lng': -79.380873}],
         'distance': 293,
         'postalCode': 'M5B 2L9',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['260 Yonge St', 'Toronto ON M5B 2L9', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1c4941735',
          'name': 'Restaurant',
          'pluralName': 'Restaurants',
          'shortName': 'Restaurant',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/default_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4e5fed38b0fb188e8c9a51e3',
        'name': 'Big Smoke Burger',
        'location': {'address': '220 Yonge St.',
         'crossStreet': 'in Urban Eatery, Toronto Eaton Centre',
         'lat': 43.65492030150413,
         'lng': -79.38080432697494,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65492030150413,
           'lng': -79.38080432697494}],
         'distance': 298,
         'postalCode': 'M5B 2H1',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['220 Yonge St. (in Urban Eatery, Toronto Eaton Centre)',
          'Toronto ON M5B 2H1',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4c87c8260dcb8cfabfd83d68',
        'name': 'Burger King',
        'location': {'address': '130 King St West',
         'crossStreet': 'in Exchange Tower',
         'lat': 43.6489639,
         'lng': -79.3836288,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.6489639,
           'lng': -79.3836288}],
         'distance': 503,
         'postalCode': 'M5X 1A6',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['130 King St West (in Exchange Tower)',
          'Toronto ON M5X 1A6',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16e941735',
          'name': 'Fast Food Restaurant',
          'pluralName': 'Fast Food Restaurants',
          'shortName': 'Fast Food',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/fastfood_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '529fad6b498eeaefffacb955',
        'name': 'South Street Burger',
        'location': {'address': '360 Bay Street',
         'crossStreet': 'Temperance',
         'lat': 43.65061154617051,
         'lng': -79.38109473310139,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65061154617051,
           'lng': -79.38109473310139}],
         'distance': 392,
         'postalCode': 'M5H 2V6',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['360 Bay Street (Temperance)',
          'Toronto ON M5H 2V6',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'venuePage': {'id': '1359971963'},
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '59b30028ee71203bd8fbb85d',
        'name': 'PLANTA Burger Financial District',
        'location': {'address': '4 Temperance St',
         'lat': 43.65093442906692,
         'lng': -79.37929871929829,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65093442906692,
           'lng': -79.37929871929829}],
         'distance': 468,
         'postalCode': 'M5H 1Y4',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['4 Temperance St', 'Toronto ON M5H 1Y4', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1d3941735',
          'name': 'Vegetarian / Vegan Restaurant',
          'pluralName': 'Vegetarian / Vegan Restaurants',
          'shortName': 'Vegetarian / Vegan',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/vegetarian_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4ae9ed9bf964a520beb721e3',
        'name': 'Burger King',
        'location': {'address': '267 College St',
         'crossStreet': 'at Spadina Ave.',
         'lat': 43.65777208307542,
         'lng': -79.39963907003408,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65777208307542,
           'lng': -79.39963907003408}],
         'distance': 1351,
         'postalCode': 'M5T 1R6',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['267 College St (at Spadina Ave.)',
          'Toronto ON M5T 1R6',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16e941735',
          'name': 'Fast Food Restaurant',
          'pluralName': 'Fast Food Restaurants',
          'shortName': 'Fast Food',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/fastfood_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4cae24479cba9521324191d9',
        'name': 'Big Smoke Burger',
        'location': {'address': '50 King St E',
         'crossStreet': 'at Toronto St.',
         'lat': 43.649722,
         'lng': -79.376206,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.649722,
           'lng': -79.376206}],
         'distance': 750,
         'postalCode': 'M5C 1E5',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['50 King St E (at Toronto St.)',
          'Toronto ON M5C 1E5',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1c4941735',
          'name': 'Restaurant',
          'pluralName': 'Restaurants',
          'shortName': 'Restaurant',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/default_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '537510e2498ebbaca9e3689b',
        'name': 'Burger King',
        'location': {'address': '243 Yonge St',
         'lat': 43.6547859,
         'lng': -79.3800973,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.6547859,
           'lng': -79.3800973}],
         'distance': 341,
         'postalCode': 'M5B 1N8',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['243 Yonge St', 'Toronto ON M5B 1N8', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16e941735',
          'name': 'Fast Food Restaurant',
          'pluralName': 'Fast Food Restaurants',
          'shortName': 'Fast Food',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/fastfood_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '544852c3498e49ed2360db96',
        'name': 'South St. Burger',
        'location': {'address': '260 King St. E',
         'lat': 43.651579049490756,
         'lng': -79.36730306874465,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.651579049490756,
           'lng': -79.36730306874465}],
         'distance': 1356,
         'postalCode': 'M5A 4L5',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['260 King St. E', 'Toronto ON M5A 4L5', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'venuePage': {'id': '1359971959'},
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '51ec6d99498e36059891dff8',
        'name': 'South St. Burger',
        'location': {'address': '1 York St #Fl 2',
         'lat': 43.64169956399765,
         'lng': -79.38027620315552,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.64169956399765,
           'lng': -79.38027620315552}],
         'distance': 1344,
         'postalCode': 'M5J 0B6',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['1 York St #Fl 2', 'Toronto ON M5J 0B6', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'venuePage': {'id': '1359971949'},
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4d8e2398d265236a0ee10217',
        'name': 'South Street Burger Co.',
        'location': {'address': '49 Clocktower Rd',
         'crossStreet': 'Shops at Don Mills',
         'lat': 43.668627,
         'lng': -79.378489,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.668627,
           'lng': -79.378489}],
         'distance': 1742,
         'postalCode': 'M3C',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['49 Clocktower Rd (Shops at Don Mills)',
          'Toronto ON M3C',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '508c0871e4b08e7b43611fd2',
        'name': 'Hero Certified Burgers',
        'location': {'address': '123 Queen Street West Unit C50',
         'lat': 43.65169377417559,
         'lng': -79.3828437661272,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65169377417559,
           'lng': -79.3828437661272}],
         'distance': 217,
         'postalCode': 'M5H 3M9',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['123 Queen Street West Unit C50',
          'Toronto ON M5H 3M9',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4b007480f964a5206c3e22e3',
        'name': 'Oh Boy Burger Market',
        'location': {'address': '571 Queen St West',
         'crossStreet': 'Portland St',
         'lat': 43.647747,
         'lng': -79.401003,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.647747,
           'lng': -79.401003}],
         'distance': 1515,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['571 Queen St West (Portland St)',
          'Toronto ON',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '5384eebb498e858f9af8d188',
        'name': 'Burger King',
        'location': {'lat': 43.65418638673085,
         'lng': -79.3799460804994,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65418638673085,
           'lng': -79.3799460804994}],
         'distance': 330,
         'cc': 'CA',
         'country': 'Canada',
         'formattedAddress': ['Canada']},
        'categories': [{'id': '4bf58dd8d48988d16e941735',
          'name': 'Fast Food Restaurant',
          'pluralName': 'Fast Food Restaurants',
          'shortName': 'Fast Food',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/fastfood_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4c72703913228cfaa4f72a65',
        'name': 'Burger King',
        'location': {'address': '555 University Ave',
         'lat': 43.657156510920736,
         'lng': -79.38667604974887,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.657156510920736,
           'lng': -79.38667604974887}],
         'distance': 464,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['555 University Ave', 'Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16e941735',
          'name': 'Fast Food Restaurant',
          'pluralName': 'Fast Food Restaurants',
          'shortName': 'Fast Food',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/fastfood_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4e9f08a777c80ac54c199abe',
        'name': 'Burgers And Fries Town',
        'location': {'address': '229 Church St.',
         'crossStreet': 'at Dundas St. E',
         'lat': 43.65657113397338,
         'lng': -79.37691083627446,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65657113397338,
           'lng': -79.37691083627446}],
         'distance': 662,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['229 Church St. (at Dundas St. E)',
          'Toronto ON',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '581f616ec25d200214d9dd24',
        'name': 'Burger Factory',
        'location': {'address': '265 Queen St W',
         'lat': 43.649969,
         'lng': -79.389145,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.649969,
           'lng': -79.389145}],
         'distance': 573,
         'postalCode': 'M5V 1Z4',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['265 Queen St W', 'Toronto ON M5V 1Z4', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4adb5d13f964a520722621e3',
        'name': 'Big Smoke Burger',
        'location': {'address': '573 King St W',
         'crossStreet': 'at Portland St.',
         'lat': 43.644385,
         'lng': -79.399567,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.644385,
           'lng': -79.399567}],
         'distance': 1615,
         'postalCode': 'M5V 1M1',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['573 King St W (at Portland St.)',
          'Toronto ON M5V 1M1',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '5823cc8cd1efa6541abcdc4d',
        'name': 'Burger Factory',
        'location': {'address': '265 Queen St W',
         'crossStreet': 'Duncan',
         'lat': 43.65031,
         'lng': -79.389127,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65031,
           'lng': -79.389127}],
         'distance': 547,
         'postalCode': 'M5V 1Z4',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['265 Queen St W (Duncan)',
          'Toronto ON M5V 1Z4',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4cfd1a0cfeec6dcbfd3a5036',
        'name': 'Hero Certified Burgers',
        'location': {'address': '200 Elizabeth Street, Unit R1A  Toronto General Hospital',
         'lat': 43.658842024763764,
         'lng': -79.3875282576804,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.658842024763764,
           'lng': -79.3875282576804}],
         'distance': 663,
         'postalCode': 'M5G 2C4',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['200 Elizabeth Street, Unit R1A  Toronto General Hospital',
          'Toronto ON M5G 2C4',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '5d4b3278c9b161000895171b',
        'name': "Jackson's Burger",
        'location': {'address': '38 Elm Street',
         'lat': 43.658509,
         'lng': -79.381589,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.658509,
           'lng': -79.381589},
          {'label': 'routing', 'lat': 43.658509, 'lng': -79.381589}],
         'distance': 590,
         'postalCode': 'M5G 2K5',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['38 Elm Street', 'Toronto ON M5G 2K5', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4baf8b68f964a52019093ce3',
        'name': 'Hero Certified Burgers',
        'location': {'lat': 43.65871085,
         'lng': -79.389667,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65871085,
           'lng': -79.389667}],
         'distance': 742,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '52e8c57c498e5826f69812d2',
        'name': "Jackson's Burger",
        'location': {'address': '374 Yonge St.',
         'crossStreet': 'at Gerrard St. W',
         'lat': 43.65879204618637,
         'lng': -79.38204478691488,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65879204618637,
           'lng': -79.38204478691488}],
         'distance': 610,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['374 Yonge St. (at Gerrard St. W)',
          'Toronto ON',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4d471afb1928a35d955fd370',
        'name': 'Burger King',
        'location': {'address': '100 King street',
         'lat': 43.682772,
         'lng': -79.419602,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.682772,
           'lng': -79.419602}],
         'distance': 4345,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['100 King street', 'Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16e941735',
          'name': 'Fast Food Restaurant',
          'pluralName': 'Fast Food Restaurants',
          'shortName': 'Fast Food',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/fastfood_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4fea0b01e4b0f5e1f943bcfa',
        'name': 'Brown Burger Truck',
        'location': {'address': '100 St George St',
         'crossStreet': 'Willcocks St',
         'lat': 43.634910505181125,
         'lng': -79.41545433816043,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.634910505181125,
           'lng': -79.41545433816043}],
         'distance': 3274,
         'postalCode': 'M5S 3G3',
         'cc': 'CA',
         'neighborhood': 'University of Toronto, Toronto, ON',
         'country': 'Canada',
         'formattedAddress': ['100 St George St (Willcocks St)',
          'M5S 3G3',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1cb941735',
          'name': 'Food Truck',
          'pluralName': 'Food Trucks',
          'shortName': 'Food Truck',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/streetfood_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '5e1caeaedf6b97000818e1f2',
        'name': 'Lobster Burger Bar',
        'location': {'address': '412 King St W',
         'lat': 43.647333,
         'lng': -79.38675,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.647333,
           'lng': -79.38675}],
         'distance': 721,
         'postalCode': 'M5V',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['412 King St W', 'Toronto ON M5V', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1c4941735',
          'name': 'Restaurant',
          'pluralName': 'Restaurants',
          'shortName': 'Restaurant',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/default_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '529fd7c8498e54bdcf391a1c',
        'name': 'South St. Burger Co.',
        'location': {'lat': 43.651912689208984,
         'lng': -79.37461853027344,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.651912689208984,
           'lng': -79.37461853027344}],
         'distance': 770,
         'cc': 'CA',
         'country': 'Canada',
         'formattedAddress': ['Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '51aa63f1498e59e83d4f9319',
        'name': "The Burger's Priest",
        'location': {'address': '463 Queen St. W.',
         'crossStreet': 'at Spadina Ave',
         'lat': 43.64838074263633,
         'lng': -79.3972643490738,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.64838074263633,
           'lng': -79.3972643490738}],
         'distance': 1214,
         'postalCode': 'M5V 2A9',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['463 Queen St. W. (at Spadina Ave)',
          'Toronto ON M5V 2A9',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16e941735',
          'name': 'Fast Food Restaurant',
          'pluralName': 'Fast Food Restaurants',
          'shortName': 'Fast Food',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/fastfood_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '5b3afd70da7080002c86b3bd',
        'name': 'Hero Certified Burgers',
        'location': {'address': '100 Queen Street West',
         'lat': 43.65364446745354,
         'lng': -79.38433170318604,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65364446745354,
           'lng': -79.38433170318604}],
         'distance': 36,
         'postalCode': 'M5H 2N2',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['100 Queen Street West',
          'Toronto ON M5H 2N2',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16c941735',
          'name': 'Burger Joint',
          'pluralName': 'Burger Joints',
          'shortName': 'Burgers',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burger_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False},
       {'id': '4b159a6af964a520fbb023e3',
        'name': 'Burger King',
        'location': {'address': '11 Leslie Street',
         'crossStreet': 'at Lake Shore Blvd. E',
         'lat': 43.65974740514479,
         'lng': -79.32833892083873,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65974740514479,
           'lng': -79.32833892083873}],
         'distance': 4531,
         'postalCode': 'M4M 3H9',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['11 Leslie Street (at Lake Shore Blvd. E)',
          'Toronto ON M4M 3H9',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16e941735',
          'name': 'Fast Food Restaurant',
          'pluralName': 'Fast Food Restaurants',
          'shortName': 'Fast Food',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/fastfood_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1612169911',
        'hasPerk': False}]}}



Get relevant part of JSON and transform it into a pandas dataframe


```python
# assign relevant part of JSON to venues
venues = results['response']['venues']

# tranform venues into a dataframe
dataframe = json_normalize(venues)
dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>categories</th>
      <th>hasPerk</th>
      <th>id</th>
      <th>location.address</th>
      <th>location.cc</th>
      <th>location.city</th>
      <th>location.country</th>
      <th>location.crossStreet</th>
      <th>location.distance</th>
      <th>location.formattedAddress</th>
      <th>location.labeledLatLngs</th>
      <th>location.lat</th>
      <th>location.lng</th>
      <th>location.neighborhood</th>
      <th>location.postalCode</th>
      <th>location.state</th>
      <th>name</th>
      <th>referralId</th>
      <th>venuePage.id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'id': '4bf58dd8d48988d1c4941735', 'name': 'R...</td>
      <td>False</td>
      <td>5b7e6b38e179100039356f91</td>
      <td>260 Yonge St</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>293</td>
      <td>[260 Yonge St, Toronto ON M5B 2L9, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.654907, 'lng':...</td>
      <td>43.654907</td>
      <td>-79.380873</td>
      <td>NaN</td>
      <td>M5B 2L9</td>
      <td>ON</td>
      <td>Big Smoke Burger</td>
      <td>v-1612169911</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'id': '4bf58dd8d48988d16c941735', 'name': 'B...</td>
      <td>False</td>
      <td>4e5fed38b0fb188e8c9a51e3</td>
      <td>220 Yonge St.</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>in Urban Eatery, Toronto Eaton Centre</td>
      <td>298</td>
      <td>[220 Yonge St. (in Urban Eatery, Toronto Eaton...</td>
      <td>[{'label': 'display', 'lat': 43.65492030150413...</td>
      <td>43.654920</td>
      <td>-79.380804</td>
      <td>NaN</td>
      <td>M5B 2H1</td>
      <td>ON</td>
      <td>Big Smoke Burger</td>
      <td>v-1612169911</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[{'id': '4bf58dd8d48988d16e941735', 'name': 'F...</td>
      <td>False</td>
      <td>4c87c8260dcb8cfabfd83d68</td>
      <td>130 King St West</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>in Exchange Tower</td>
      <td>503</td>
      <td>[130 King St West (in Exchange Tower), Toronto...</td>
      <td>[{'label': 'display', 'lat': 43.6489639, 'lng'...</td>
      <td>43.648964</td>
      <td>-79.383629</td>
      <td>NaN</td>
      <td>M5X 1A6</td>
      <td>ON</td>
      <td>Burger King</td>
      <td>v-1612169911</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[{'id': '4bf58dd8d48988d16c941735', 'name': 'B...</td>
      <td>False</td>
      <td>529fad6b498eeaefffacb955</td>
      <td>360 Bay Street</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>Temperance</td>
      <td>392</td>
      <td>[360 Bay Street (Temperance), Toronto ON M5H 2...</td>
      <td>[{'label': 'display', 'lat': 43.65061154617051...</td>
      <td>43.650612</td>
      <td>-79.381095</td>
      <td>NaN</td>
      <td>M5H 2V6</td>
      <td>ON</td>
      <td>South Street Burger</td>
      <td>v-1612169911</td>
      <td>1359971963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[{'id': '4bf58dd8d48988d1d3941735', 'name': 'V...</td>
      <td>False</td>
      <td>59b30028ee71203bd8fbb85d</td>
      <td>4 Temperance St</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>468</td>
      <td>[4 Temperance St, Toronto ON M5H 1Y4, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.65093442906692...</td>
      <td>43.650934</td>
      <td>-79.379299</td>
      <td>NaN</td>
      <td>M5H 1Y4</td>
      <td>ON</td>
      <td>PLANTA Burger Financial District</td>
      <td>v-1612169911</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Define information of interest and filter dataframe


```python
# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

dataframe_filtered
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>categories</th>
      <th>address</th>
      <th>cc</th>
      <th>city</th>
      <th>country</th>
      <th>crossStreet</th>
      <th>distance</th>
      <th>formattedAddress</th>
      <th>labeledLatLngs</th>
      <th>lat</th>
      <th>lng</th>
      <th>neighborhood</th>
      <th>postalCode</th>
      <th>state</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Big Smoke Burger</td>
      <td>Restaurant</td>
      <td>260 Yonge St</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>293</td>
      <td>[260 Yonge St, Toronto ON M5B 2L9, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.654907, 'lng':...</td>
      <td>43.654907</td>
      <td>-79.380873</td>
      <td>NaN</td>
      <td>M5B 2L9</td>
      <td>ON</td>
      <td>5b7e6b38e179100039356f91</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Big Smoke Burger</td>
      <td>Burger Joint</td>
      <td>220 Yonge St.</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>in Urban Eatery, Toronto Eaton Centre</td>
      <td>298</td>
      <td>[220 Yonge St. (in Urban Eatery, Toronto Eaton...</td>
      <td>[{'label': 'display', 'lat': 43.65492030150413...</td>
      <td>43.654920</td>
      <td>-79.380804</td>
      <td>NaN</td>
      <td>M5B 2H1</td>
      <td>ON</td>
      <td>4e5fed38b0fb188e8c9a51e3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Burger King</td>
      <td>Fast Food Restaurant</td>
      <td>130 King St West</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>in Exchange Tower</td>
      <td>503</td>
      <td>[130 King St West (in Exchange Tower), Toronto...</td>
      <td>[{'label': 'display', 'lat': 43.6489639, 'lng'...</td>
      <td>43.648964</td>
      <td>-79.383629</td>
      <td>NaN</td>
      <td>M5X 1A6</td>
      <td>ON</td>
      <td>4c87c8260dcb8cfabfd83d68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>South Street Burger</td>
      <td>Burger Joint</td>
      <td>360 Bay Street</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>Temperance</td>
      <td>392</td>
      <td>[360 Bay Street (Temperance), Toronto ON M5H 2...</td>
      <td>[{'label': 'display', 'lat': 43.65061154617051...</td>
      <td>43.650612</td>
      <td>-79.381095</td>
      <td>NaN</td>
      <td>M5H 2V6</td>
      <td>ON</td>
      <td>529fad6b498eeaefffacb955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PLANTA Burger Financial District</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>4 Temperance St</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>468</td>
      <td>[4 Temperance St, Toronto ON M5H 1Y4, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.65093442906692...</td>
      <td>43.650934</td>
      <td>-79.379299</td>
      <td>NaN</td>
      <td>M5H 1Y4</td>
      <td>ON</td>
      <td>59b30028ee71203bd8fbb85d</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Burger King</td>
      <td>Fast Food Restaurant</td>
      <td>267 College St</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>at Spadina Ave.</td>
      <td>1351</td>
      <td>[267 College St (at Spadina Ave.), Toronto ON ...</td>
      <td>[{'label': 'display', 'lat': 43.65777208307542...</td>
      <td>43.657772</td>
      <td>-79.399639</td>
      <td>NaN</td>
      <td>M5T 1R6</td>
      <td>ON</td>
      <td>4ae9ed9bf964a520beb721e3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Big Smoke Burger</td>
      <td>Restaurant</td>
      <td>50 King St E</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>at Toronto St.</td>
      <td>750</td>
      <td>[50 King St E (at Toronto St.), Toronto ON M5C...</td>
      <td>[{'label': 'display', 'lat': 43.649722, 'lng':...</td>
      <td>43.649722</td>
      <td>-79.376206</td>
      <td>NaN</td>
      <td>M5C 1E5</td>
      <td>ON</td>
      <td>4cae24479cba9521324191d9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Burger King</td>
      <td>Fast Food Restaurant</td>
      <td>243 Yonge St</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>341</td>
      <td>[243 Yonge St, Toronto ON M5B 1N8, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.6547859, 'lng'...</td>
      <td>43.654786</td>
      <td>-79.380097</td>
      <td>NaN</td>
      <td>M5B 1N8</td>
      <td>ON</td>
      <td>537510e2498ebbaca9e3689b</td>
    </tr>
    <tr>
      <th>8</th>
      <td>South St. Burger</td>
      <td>Burger Joint</td>
      <td>260 King St. E</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>1356</td>
      <td>[260 King St. E, Toronto ON M5A 4L5, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.65157904949075...</td>
      <td>43.651579</td>
      <td>-79.367303</td>
      <td>NaN</td>
      <td>M5A 4L5</td>
      <td>ON</td>
      <td>544852c3498e49ed2360db96</td>
    </tr>
    <tr>
      <th>9</th>
      <td>South St. Burger</td>
      <td>Burger Joint</td>
      <td>1 York St #Fl 2</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>1344</td>
      <td>[1 York St #Fl 2, Toronto ON M5J 0B6, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.64169956399765...</td>
      <td>43.641700</td>
      <td>-79.380276</td>
      <td>NaN</td>
      <td>M5J 0B6</td>
      <td>ON</td>
      <td>51ec6d99498e36059891dff8</td>
    </tr>
    <tr>
      <th>10</th>
      <td>South Street Burger Co.</td>
      <td>Burger Joint</td>
      <td>49 Clocktower Rd</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>Shops at Don Mills</td>
      <td>1742</td>
      <td>[49 Clocktower Rd (Shops at Don Mills), Toront...</td>
      <td>[{'label': 'display', 'lat': 43.668627, 'lng':...</td>
      <td>43.668627</td>
      <td>-79.378489</td>
      <td>NaN</td>
      <td>M3C</td>
      <td>ON</td>
      <td>4d8e2398d265236a0ee10217</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hero Certified Burgers</td>
      <td>Burger Joint</td>
      <td>123 Queen Street West Unit C50</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>217</td>
      <td>[123 Queen Street West Unit C50, Toronto ON M5...</td>
      <td>[{'label': 'display', 'lat': 43.65169377417559...</td>
      <td>43.651694</td>
      <td>-79.382844</td>
      <td>NaN</td>
      <td>M5H 3M9</td>
      <td>ON</td>
      <td>508c0871e4b08e7b43611fd2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Oh Boy Burger Market</td>
      <td>Burger Joint</td>
      <td>571 Queen St West</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>Portland St</td>
      <td>1515</td>
      <td>[571 Queen St West (Portland St), Toronto ON, ...</td>
      <td>[{'label': 'display', 'lat': 43.647747, 'lng':...</td>
      <td>43.647747</td>
      <td>-79.401003</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ON</td>
      <td>4b007480f964a5206c3e22e3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Burger King</td>
      <td>Fast Food Restaurant</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>330</td>
      <td>[Canada]</td>
      <td>[{'label': 'display', 'lat': 43.65418638673085...</td>
      <td>43.654186</td>
      <td>-79.379946</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5384eebb498e858f9af8d188</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Burger King</td>
      <td>Fast Food Restaurant</td>
      <td>555 University Ave</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>464</td>
      <td>[555 University Ave, Toronto ON, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.65715651092073...</td>
      <td>43.657157</td>
      <td>-79.386676</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ON</td>
      <td>4c72703913228cfaa4f72a65</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Burgers And Fries Town</td>
      <td>Burger Joint</td>
      <td>229 Church St.</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>at Dundas St. E</td>
      <td>662</td>
      <td>[229 Church St. (at Dundas St. E), Toronto ON,...</td>
      <td>[{'label': 'display', 'lat': 43.65657113397338...</td>
      <td>43.656571</td>
      <td>-79.376911</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ON</td>
      <td>4e9f08a777c80ac54c199abe</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Burger Factory</td>
      <td>Burger Joint</td>
      <td>265 Queen St W</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>573</td>
      <td>[265 Queen St W, Toronto ON M5V 1Z4, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.649969, 'lng':...</td>
      <td>43.649969</td>
      <td>-79.389145</td>
      <td>NaN</td>
      <td>M5V 1Z4</td>
      <td>ON</td>
      <td>581f616ec25d200214d9dd24</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Big Smoke Burger</td>
      <td>Burger Joint</td>
      <td>573 King St W</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>at Portland St.</td>
      <td>1615</td>
      <td>[573 King St W (at Portland St.), Toronto ON M...</td>
      <td>[{'label': 'display', 'lat': 43.644385, 'lng':...</td>
      <td>43.644385</td>
      <td>-79.399567</td>
      <td>NaN</td>
      <td>M5V 1M1</td>
      <td>ON</td>
      <td>4adb5d13f964a520722621e3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Burger Factory</td>
      <td>Burger Joint</td>
      <td>265 Queen St W</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>Duncan</td>
      <td>547</td>
      <td>[265 Queen St W (Duncan), Toronto ON M5V 1Z4, ...</td>
      <td>[{'label': 'display', 'lat': 43.65031, 'lng': ...</td>
      <td>43.650310</td>
      <td>-79.389127</td>
      <td>NaN</td>
      <td>M5V 1Z4</td>
      <td>ON</td>
      <td>5823cc8cd1efa6541abcdc4d</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Hero Certified Burgers</td>
      <td>Burger Joint</td>
      <td>200 Elizabeth Street, Unit R1A  Toronto Genera...</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>663</td>
      <td>[200 Elizabeth Street, Unit R1A  Toronto Gener...</td>
      <td>[{'label': 'display', 'lat': 43.65884202476376...</td>
      <td>43.658842</td>
      <td>-79.387528</td>
      <td>NaN</td>
      <td>M5G 2C4</td>
      <td>ON</td>
      <td>4cfd1a0cfeec6dcbfd3a5036</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Jackson's Burger</td>
      <td>Burger Joint</td>
      <td>38 Elm Street</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>590</td>
      <td>[38 Elm Street, Toronto ON M5G 2K5, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.658509, 'lng':...</td>
      <td>43.658509</td>
      <td>-79.381589</td>
      <td>NaN</td>
      <td>M5G 2K5</td>
      <td>ON</td>
      <td>5d4b3278c9b161000895171b</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Hero Certified Burgers</td>
      <td>Burger Joint</td>
      <td>NaN</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>742</td>
      <td>[Toronto ON, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.65871085, 'lng...</td>
      <td>43.658711</td>
      <td>-79.389667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ON</td>
      <td>4baf8b68f964a52019093ce3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Jackson's Burger</td>
      <td>Burger Joint</td>
      <td>374 Yonge St.</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>at Gerrard St. W</td>
      <td>610</td>
      <td>[374 Yonge St. (at Gerrard St. W), Toronto ON,...</td>
      <td>[{'label': 'display', 'lat': 43.65879204618637...</td>
      <td>43.658792</td>
      <td>-79.382045</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ON</td>
      <td>52e8c57c498e5826f69812d2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Burger King</td>
      <td>Fast Food Restaurant</td>
      <td>100 King street</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>4345</td>
      <td>[100 King street, Toronto ON, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.682772, 'lng':...</td>
      <td>43.682772</td>
      <td>-79.419602</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ON</td>
      <td>4d471afb1928a35d955fd370</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Brown Burger Truck</td>
      <td>Food Truck</td>
      <td>100 St George St</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Canada</td>
      <td>Willcocks St</td>
      <td>3274</td>
      <td>[100 St George St (Willcocks St), M5S 3G3, Can...</td>
      <td>[{'label': 'display', 'lat': 43.63491050518112...</td>
      <td>43.634911</td>
      <td>-79.415454</td>
      <td>University of Toronto, Toronto, ON</td>
      <td>M5S 3G3</td>
      <td>NaN</td>
      <td>4fea0b01e4b0f5e1f943bcfa</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Lobster Burger Bar</td>
      <td>Restaurant</td>
      <td>412 King St W</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>721</td>
      <td>[412 King St W, Toronto ON M5V, Canada]</td>
      <td>[{'label': 'display', 'lat': 43.647333, 'lng':...</td>
      <td>43.647333</td>
      <td>-79.386750</td>
      <td>NaN</td>
      <td>M5V</td>
      <td>ON</td>
      <td>5e1caeaedf6b97000818e1f2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>South St. Burger Co.</td>
      <td>Burger Joint</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>770</td>
      <td>[Canada]</td>
      <td>[{'label': 'display', 'lat': 43.65191268920898...</td>
      <td>43.651913</td>
      <td>-79.374619</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>529fd7c8498e54bdcf391a1c</td>
    </tr>
    <tr>
      <th>27</th>
      <td>The Burger's Priest</td>
      <td>Fast Food Restaurant</td>
      <td>463 Queen St. W.</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>at Spadina Ave</td>
      <td>1214</td>
      <td>[463 Queen St. W. (at Spadina Ave), Toronto ON...</td>
      <td>[{'label': 'display', 'lat': 43.64838074263633...</td>
      <td>43.648381</td>
      <td>-79.397264</td>
      <td>NaN</td>
      <td>M5V 2A9</td>
      <td>ON</td>
      <td>51aa63f1498e59e83d4f9319</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Hero Certified Burgers</td>
      <td>Burger Joint</td>
      <td>100 Queen Street West</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>NaN</td>
      <td>36</td>
      <td>[100 Queen Street West, Toronto ON M5H 2N2, Ca...</td>
      <td>[{'label': 'display', 'lat': 43.65364446745354...</td>
      <td>43.653644</td>
      <td>-79.384332</td>
      <td>NaN</td>
      <td>M5H 2N2</td>
      <td>ON</td>
      <td>5b3afd70da7080002c86b3bd</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Burger King</td>
      <td>Fast Food Restaurant</td>
      <td>11 Leslie Street</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>Canada</td>
      <td>at Lake Shore Blvd. E</td>
      <td>4531</td>
      <td>[11 Leslie Street (at Lake Shore Blvd. E), Tor...</td>
      <td>[{'label': 'display', 'lat': 43.65974740514479...</td>
      <td>43.659747</td>
      <td>-79.328339</td>
      <td>NaN</td>
      <td>M4M 3H9</td>
      <td>ON</td>
      <td>4b159a6af964a520fbb023e3</td>
    </tr>
  </tbody>
</table>
</div>



Let's visualize the Burger restaurants that are nearby


```python
dataframe_filtered.name
```




    0                     Big Smoke Burger
    1                     Big Smoke Burger
    2                          Burger King
    3                  South Street Burger
    4     PLANTA Burger Financial District
    5                          Burger King
    6                     Big Smoke Burger
    7                          Burger King
    8                     South St. Burger
    9                     South St. Burger
    10             South Street Burger Co.
    11              Hero Certified Burgers
    12                Oh Boy Burger Market
    13                         Burger King
    14                         Burger King
    15              Burgers And Fries Town
    16                      Burger Factory
    17                    Big Smoke Burger
    18                      Burger Factory
    19              Hero Certified Burgers
    20                    Jackson's Burger
    21              Hero Certified Burgers
    22                    Jackson's Burger
    23                         Burger King
    24                  Brown Burger Truck
    25                  Lobster Burger Bar
    26                South St. Burger Co.
    27                 The Burger's Priest
    28              Hero Certified Burgers
    29                         Burger King
    Name: name, dtype: object




```python
venues_map = folium.Map(location=[latitude, longitude], zoom_start=13) # generate map centred around the Conrad Hotel

# add a red circle marker to represent the Conrad Hotel
folium.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Conrad Hotel',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

# add the Italian restaurants as blue circle markers
for lat, lng, label in zip(dataframe_filtered.lat, dataframe_filtered.lng, dataframe_filtered.categories):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)

# display map
venues_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css%22/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_9c78b5aa03b2493baf3bde1c35104dce%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_9c78b5aa03b2493baf3bde1c35104dce%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_9c78b5aa03b2493baf3bde1c35104dce%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_9c78b5aa03b2493baf3bde1c35104dce%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B43.6534817%2C%20-79.3839347%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2013%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_1776799f901348ccb175c59f9c8b3d5c%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_68c12c64bb864764aaf0877b77b4e3e1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6534817%2C%20-79.3839347%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22red%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22red%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%2010%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a454931e73bc4de6aa6c3a095c68be3f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2469378cffb14456998ef0459a118dc2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2469378cffb14456998ef0459a118dc2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EConrad%20Hotel%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a454931e73bc4de6aa6c3a095c68be3f.setContent%28html_2469378cffb14456998ef0459a118dc2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_68c12c64bb864764aaf0877b77b4e3e1.bindPopup%28popup_a454931e73bc4de6aa6c3a095c68be3f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2f9547eeb1ab4976b33410cb418ee092%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.654907%2C%20-79.380873%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3a754af75ea14736b009b8d75dcbeb72%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_513c417d538e4da7b4f33521ee150ca6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_513c417d538e4da7b4f33521ee150ca6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERestaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3a754af75ea14736b009b8d75dcbeb72.setContent%28html_513c417d538e4da7b4f33521ee150ca6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2f9547eeb1ab4976b33410cb418ee092.bindPopup%28popup_3a754af75ea14736b009b8d75dcbeb72%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_62d6d9a03c5b4a8aa1739d190ae753b2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65492030150413%2C%20-79.38080432697494%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_fe1bcd53786e40f5856fc3e68da1778f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_84ba7e1028524df98c99cc22e958bcf3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_84ba7e1028524df98c99cc22e958bcf3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_fe1bcd53786e40f5856fc3e68da1778f.setContent%28html_84ba7e1028524df98c99cc22e958bcf3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_62d6d9a03c5b4a8aa1739d190ae753b2.bindPopup%28popup_fe1bcd53786e40f5856fc3e68da1778f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_70d6deed77904430adbe322491b59f0f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6489639%2C%20-79.3836288%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3ab2eb4309af452c843b1c1b32b1390c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9f81616153434b32ac58dc6028012786%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9f81616153434b32ac58dc6028012786%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFast%20Food%20Restaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3ab2eb4309af452c843b1c1b32b1390c.setContent%28html_9f81616153434b32ac58dc6028012786%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_70d6deed77904430adbe322491b59f0f.bindPopup%28popup_3ab2eb4309af452c843b1c1b32b1390c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_29f7c83f3ed645788fabd01b17cf0f4a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65061154617051%2C%20-79.38109473310139%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_bbb675d8a4f145aaa02da5b8d1bfb8ea%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d9b2e03b0c4a42c0a0bd7f9c63aa9254%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d9b2e03b0c4a42c0a0bd7f9c63aa9254%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_bbb675d8a4f145aaa02da5b8d1bfb8ea.setContent%28html_d9b2e03b0c4a42c0a0bd7f9c63aa9254%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_29f7c83f3ed645788fabd01b17cf0f4a.bindPopup%28popup_bbb675d8a4f145aaa02da5b8d1bfb8ea%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ee70ef16539f4e56bc2e1b5eb423a860%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65093442906692%2C%20-79.37929871929829%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_16f7ae6d0fb04a2f9c37b57acf818b17%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ddbdb3c710634f67acec053e989b0048%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ddbdb3c710634f67acec053e989b0048%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EVegetarian%20/%20Vegan%20Restaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_16f7ae6d0fb04a2f9c37b57acf818b17.setContent%28html_ddbdb3c710634f67acec053e989b0048%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ee70ef16539f4e56bc2e1b5eb423a860.bindPopup%28popup_16f7ae6d0fb04a2f9c37b57acf818b17%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3f7e1a8f4ba04501bc368f8f6fcd26b2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65777208307542%2C%20-79.39963907003408%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b2638e91e71741fa8339a471e1f2172d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2b85efcdf639423c8d422aad5b68453e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2b85efcdf639423c8d422aad5b68453e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFast%20Food%20Restaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b2638e91e71741fa8339a471e1f2172d.setContent%28html_2b85efcdf639423c8d422aad5b68453e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3f7e1a8f4ba04501bc368f8f6fcd26b2.bindPopup%28popup_b2638e91e71741fa8339a471e1f2172d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_82ff80f4933e469e88efc82469413221%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.649722%2C%20-79.376206%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_83e1a30580b44e96b6054507f6e3ca4a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a2e082f932994f6bb3be76da7af4dd24%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a2e082f932994f6bb3be76da7af4dd24%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERestaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_83e1a30580b44e96b6054507f6e3ca4a.setContent%28html_a2e082f932994f6bb3be76da7af4dd24%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_82ff80f4933e469e88efc82469413221.bindPopup%28popup_83e1a30580b44e96b6054507f6e3ca4a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7a1ddd813812440a99951d6c6506d8e4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6547859%2C%20-79.3800973%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c990c295e5d0448b8488769953d43269%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_958c4fa4599a4fbfa3814e3247576e14%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_958c4fa4599a4fbfa3814e3247576e14%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFast%20Food%20Restaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c990c295e5d0448b8488769953d43269.setContent%28html_958c4fa4599a4fbfa3814e3247576e14%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7a1ddd813812440a99951d6c6506d8e4.bindPopup%28popup_c990c295e5d0448b8488769953d43269%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4c75f3ef716445819f588d20c1e25bb9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.651579049490756%2C%20-79.36730306874465%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_141653e783ec41e7be501272ff7cc973%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_58b871abce72472591e59d27e1576544%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_58b871abce72472591e59d27e1576544%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_141653e783ec41e7be501272ff7cc973.setContent%28html_58b871abce72472591e59d27e1576544%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4c75f3ef716445819f588d20c1e25bb9.bindPopup%28popup_141653e783ec41e7be501272ff7cc973%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fc43c4875746437d9a3793aebc8f5fd9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.64169956399765%2C%20-79.38027620315552%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_047a7ff1c20b43488a6ecccc04167eb2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3c722262397f4a2db4a93a5b7dd90811%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3c722262397f4a2db4a93a5b7dd90811%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_047a7ff1c20b43488a6ecccc04167eb2.setContent%28html_3c722262397f4a2db4a93a5b7dd90811%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fc43c4875746437d9a3793aebc8f5fd9.bindPopup%28popup_047a7ff1c20b43488a6ecccc04167eb2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ce20625c33ba440c99aa89bbf7983393%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.668627%2C%20-79.378489%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8d6ca772150445449d99e565c094f203%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_20f7553e06ca469482857e2d4da87921%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_20f7553e06ca469482857e2d4da87921%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8d6ca772150445449d99e565c094f203.setContent%28html_20f7553e06ca469482857e2d4da87921%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ce20625c33ba440c99aa89bbf7983393.bindPopup%28popup_8d6ca772150445449d99e565c094f203%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2e5b1e14ddd349d7b0e3f41cac97ea56%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65169377417559%2C%20-79.3828437661272%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_80eea3eeaaa1428f99a02dfb69019740%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b20d813947a04a08977ef20e32548e34%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b20d813947a04a08977ef20e32548e34%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_80eea3eeaaa1428f99a02dfb69019740.setContent%28html_b20d813947a04a08977ef20e32548e34%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2e5b1e14ddd349d7b0e3f41cac97ea56.bindPopup%28popup_80eea3eeaaa1428f99a02dfb69019740%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_98caf494e9e14dd4840107184a451222%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.647747%2C%20-79.401003%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7e360261238147b8b44b85f1ffbe56ed%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7819b11b9fe8491b85fcd91e5d86a14b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7819b11b9fe8491b85fcd91e5d86a14b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7e360261238147b8b44b85f1ffbe56ed.setContent%28html_7819b11b9fe8491b85fcd91e5d86a14b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_98caf494e9e14dd4840107184a451222.bindPopup%28popup_7e360261238147b8b44b85f1ffbe56ed%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b0eb595af60844f8b880c81718cbe905%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65418638673085%2C%20-79.3799460804994%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f213d83dd6ff4828a52f809929af77f4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e9049565f8c5429e85535c570719d560%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e9049565f8c5429e85535c570719d560%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFast%20Food%20Restaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f213d83dd6ff4828a52f809929af77f4.setContent%28html_e9049565f8c5429e85535c570719d560%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b0eb595af60844f8b880c81718cbe905.bindPopup%28popup_f213d83dd6ff4828a52f809929af77f4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0650a6ec4a3749a5af08364c93a5f298%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.657156510920736%2C%20-79.38667604974887%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f134c2e12cfa4d98bd0697d3ead31afc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5ee7e249aba446d9ba034c929b91cf79%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5ee7e249aba446d9ba034c929b91cf79%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFast%20Food%20Restaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f134c2e12cfa4d98bd0697d3ead31afc.setContent%28html_5ee7e249aba446d9ba034c929b91cf79%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0650a6ec4a3749a5af08364c93a5f298.bindPopup%28popup_f134c2e12cfa4d98bd0697d3ead31afc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_945d9e6e5670439ebfa2202526ddba44%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65657113397338%2C%20-79.37691083627446%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3b424deaa68042c5873e674cacf7343e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_88a9937f0d1744848d62f04f26a5a90e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_88a9937f0d1744848d62f04f26a5a90e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3b424deaa68042c5873e674cacf7343e.setContent%28html_88a9937f0d1744848d62f04f26a5a90e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_945d9e6e5670439ebfa2202526ddba44.bindPopup%28popup_3b424deaa68042c5873e674cacf7343e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fc41cca06f1d4d85acbe5b18626d5fea%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.649969%2C%20-79.389145%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5445375255fb411c850ada9c2069e485%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f39c4aa189094f6aab0eceba753bacc5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f39c4aa189094f6aab0eceba753bacc5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5445375255fb411c850ada9c2069e485.setContent%28html_f39c4aa189094f6aab0eceba753bacc5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fc41cca06f1d4d85acbe5b18626d5fea.bindPopup%28popup_5445375255fb411c850ada9c2069e485%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fb130ae676454b90b9f8eb43a33054f4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.644385%2C%20-79.399567%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_465fbe42e0e7484e8bf11542ba931850%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cbac66a5339140148ab2cc71f86fb296%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_cbac66a5339140148ab2cc71f86fb296%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_465fbe42e0e7484e8bf11542ba931850.setContent%28html_cbac66a5339140148ab2cc71f86fb296%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fb130ae676454b90b9f8eb43a33054f4.bindPopup%28popup_465fbe42e0e7484e8bf11542ba931850%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a35a67487a18436b847071550e3eaef2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65031%2C%20-79.389127%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_898834f5bb5244448d9c4f9af0a7ae4b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1f8d4818274348038ed1095f30571366%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1f8d4818274348038ed1095f30571366%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_898834f5bb5244448d9c4f9af0a7ae4b.setContent%28html_1f8d4818274348038ed1095f30571366%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a35a67487a18436b847071550e3eaef2.bindPopup%28popup_898834f5bb5244448d9c4f9af0a7ae4b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_41bf447e6b8446ce87212374fa25434d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.658842024763764%2C%20-79.3875282576804%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1348abe4d1ae4eb3905497050cf19cbf%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_60be3e39d6474e10a67bac39d06ae7a4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_60be3e39d6474e10a67bac39d06ae7a4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1348abe4d1ae4eb3905497050cf19cbf.setContent%28html_60be3e39d6474e10a67bac39d06ae7a4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_41bf447e6b8446ce87212374fa25434d.bindPopup%28popup_1348abe4d1ae4eb3905497050cf19cbf%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_28d4befacdb94b1d84487025f3889299%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.658509%2C%20-79.381589%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_983eac9cb8984016b369a1b4d0a98c78%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8b392bc690524ca59017a33e1e927a76%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8b392bc690524ca59017a33e1e927a76%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_983eac9cb8984016b369a1b4d0a98c78.setContent%28html_8b392bc690524ca59017a33e1e927a76%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_28d4befacdb94b1d84487025f3889299.bindPopup%28popup_983eac9cb8984016b369a1b4d0a98c78%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b388bad77332484e98eda7e1c217b1fd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65871085%2C%20-79.389667%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d6102bd8c40f482a8325868ad24dd923%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2389f75af6f54f9498510ec8e3947f09%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2389f75af6f54f9498510ec8e3947f09%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d6102bd8c40f482a8325868ad24dd923.setContent%28html_2389f75af6f54f9498510ec8e3947f09%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b388bad77332484e98eda7e1c217b1fd.bindPopup%28popup_d6102bd8c40f482a8325868ad24dd923%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7b53d34ef2654b1dbb6481ba19accf3b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65879204618637%2C%20-79.38204478691488%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_98871a6b22cd4b0db2f0d5a86e2a2f5b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b980dc510ecd4020bbfc08a4bff69897%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b980dc510ecd4020bbfc08a4bff69897%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_98871a6b22cd4b0db2f0d5a86e2a2f5b.setContent%28html_b980dc510ecd4020bbfc08a4bff69897%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7b53d34ef2654b1dbb6481ba19accf3b.bindPopup%28popup_98871a6b22cd4b0db2f0d5a86e2a2f5b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6de5ee04c6cb4e368af881a07229b6f0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.682772%2C%20-79.419602%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1ceaa2b172194a608db96cfb6e456b63%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dfc8836a8fc940b59faef5da4087d6e6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dfc8836a8fc940b59faef5da4087d6e6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFast%20Food%20Restaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1ceaa2b172194a608db96cfb6e456b63.setContent%28html_dfc8836a8fc940b59faef5da4087d6e6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6de5ee04c6cb4e368af881a07229b6f0.bindPopup%28popup_1ceaa2b172194a608db96cfb6e456b63%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_380c5e9644e34c9f81e069a167929acb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.634910505181125%2C%20-79.41545433816043%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8247f17a4d3e4cd3af4b0c6b89842399%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_02712f246837403094bd41fb42b37fd1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_02712f246837403094bd41fb42b37fd1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFood%20Truck%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8247f17a4d3e4cd3af4b0c6b89842399.setContent%28html_02712f246837403094bd41fb42b37fd1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_380c5e9644e34c9f81e069a167929acb.bindPopup%28popup_8247f17a4d3e4cd3af4b0c6b89842399%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d1e9c2ebfa654a209b6f6fe1c4f5e882%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.647333%2C%20-79.38675%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_33a069f5e8274b068fc4449f6f5fac9e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_74074b0dfdb8424fb5b9678866324fd0%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_74074b0dfdb8424fb5b9678866324fd0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERestaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_33a069f5e8274b068fc4449f6f5fac9e.setContent%28html_74074b0dfdb8424fb5b9678866324fd0%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d1e9c2ebfa654a209b6f6fe1c4f5e882.bindPopup%28popup_33a069f5e8274b068fc4449f6f5fac9e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4264786875cc4f3fbd54c77a5e7e26f2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.651912689208984%2C%20-79.37461853027344%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_67716398e0524c7cbac6876a07b43a09%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d53657f5e4cc4513ae5e58da210996d3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d53657f5e4cc4513ae5e58da210996d3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_67716398e0524c7cbac6876a07b43a09.setContent%28html_d53657f5e4cc4513ae5e58da210996d3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4264786875cc4f3fbd54c77a5e7e26f2.bindPopup%28popup_67716398e0524c7cbac6876a07b43a09%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_92698e8d1c3c4913bd1c751cd33b3b42%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.64838074263633%2C%20-79.3972643490738%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a199c833bd1e47009511492e2628f968%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dd8e3eaab57c441db6319ac7cb50ab53%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dd8e3eaab57c441db6319ac7cb50ab53%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFast%20Food%20Restaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a199c833bd1e47009511492e2628f968.setContent%28html_dd8e3eaab57c441db6319ac7cb50ab53%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_92698e8d1c3c4913bd1c751cd33b3b42.bindPopup%28popup_a199c833bd1e47009511492e2628f968%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9bd9a50776f544ee94c6b8e74a14384c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65364446745354%2C%20-79.38433170318604%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4e5b643bd18a4dd3ac59309d1f5edba3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e964bf7357c74bd6a28c6df58c7a8f0b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e964bf7357c74bd6a28c6df58c7a8f0b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBurger%20Joint%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4e5b643bd18a4dd3ac59309d1f5edba3.setContent%28html_e964bf7357c74bd6a28c6df58c7a8f0b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9bd9a50776f544ee94c6b8e74a14384c.bindPopup%28popup_4e5b643bd18a4dd3ac59309d1f5edba3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ee543c307a224e35968b93c9bcfe2955%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65974740514479%2C%20-79.32833892083873%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22blue%22%2C%20%22fillOpacity%22%3A%200.6%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_9c78b5aa03b2493baf3bde1c35104dce%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_414be7944e6a4ed9bf4c488dcec6f4bf%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9e8886aa73c84bbd80049fc43123766c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9e8886aa73c84bbd80049fc43123766c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFast%20Food%20Restaurant%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_414be7944e6a4ed9bf4c488dcec6f4bf.setContent%28html_9e8886aa73c84bbd80049fc43123766c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ee543c307a224e35968b93c9bcfe2955.bindPopup%28popup_414be7944e6a4ed9bf4c488dcec6f4bf%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python

```
