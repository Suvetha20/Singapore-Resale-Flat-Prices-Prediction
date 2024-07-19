import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# ----------------------------------------Functions---------------------------------------------------------
# To get mrt_df
def get_mrt_df():
    df = pd.read_csv('mrt.csv').reset_index(drop= True)
    return df

# To load pickles
def load_pickles():
    with open(r"model.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    with open(r'scaler.pkl', 'rb') as f:
        scaler_loaded = pickle.load(f)
    return loaded_model, scaler_loaded
# Pre_processing input    
def preprocessing_input(street_name,block,lease_commence_year,storey_range):
    # -----Calculating lease_remain_years using lease_commence_year-----
    lease_remain_years = 99 - (2023 - lease_commence_year)

    # -----Calculating median of storey_range to make our calculations quite comfortable-----
    split_list = storey_range.split(' TO ')
    float_list = [float(i) for i in split_list]
    storey_median = statistics.median(float_list)
    min_dist_mrt, cbd_dist = get_min_distance_from_MRT_CBD(block,street_name)
    return storey_median,lease_remain_years, min_dist_mrt, cbd_dist

def get_min_distance_from_MRT_CBD(block,street_name):

    # Getting the address by joining the block number and the street name
    addrs = block + " " + street_name
    url = "https://www.onemap.gov.sg/api/common/elastic/search?searchVal="+str(addrs)+"&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    resp = requests.get(url)

    # Using OpenMap API getting the latitude and longitude location of that address
    origin = []
    data_geo_location = json.loads(resp.content)
    if data_geo_location['found'] != 0:
            latitude = data_geo_location['results'][0]['LATITUDE']
            longitude = data_geo_location['results'][0]['LONGITUDE']
            origin.append((latitude, longitude))
    else:
        st.warning('Location Not found')
        return None

    # Appending the Latitudes and Longitudes of the MRT Stations
    # Latitudes and Longitudes are been appended in the form of a tuple to that list
    mrt_location = get_mrt_df()
    mrt_lat = mrt_location['latitude']
    mrt_long = mrt_location['longitude']
    list_of_mrt_coordinates = []
    for lat, long in zip(mrt_lat, mrt_long):
        list_of_mrt_coordinates.append((lat, long))

    # Getting distance to nearest MRT Stations (Mass Rapid Transit System)
    list_of_dist_mrt = []
    for destination in range(0, len(list_of_mrt_coordinates)):
        list_of_dist_mrt.append(geodesic(origin, list_of_mrt_coordinates[destination]).meters)
    min_dist_mrt = min(list_of_dist_mrt)

    # Getting distance from CBD (Central Business District)
    cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates\
    
    return min_dist_mrt,cbd_dist


# -------------------------------This is the configuration page for our Streamlit Application---------------------------

st.set_page_config(layout="wide",
                   page_icon="🏨")

st.title(":rainbow[SINGAPORE RESALE FLAT PRICES PREDICTING]")
st.write("")
# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )

# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.header(":blue[HDB Flats:]")
    st.write('''The majority of Singaporeans live in public housing provided by the HDB.
    HDB flats can be purchased either directly from the HDB as a new unit or through the resale market from existing owners.''')
    
    st.header(":blue[Resale Process:]")
    st.write('''In the resale market, buyers purchase flats from existing flat owners, and the transactions are facilitated through the HDB resale process.
    The process involves a series of steps, including valuation, negotiations, and the submission of necessary documents.''')
    
    st.header(":blue[Valuation:]")
    st.write('''The HDB conducts a valuation of the flat to determine its market value. This is important for both buyers and sellers in negotiating a fair price.''')
    
    st.header(":blue[Eligibility Criteria:]")
    st.write("Buyers and sellers in the resale market must meet certain eligibility criteria, including citizenship requirements and income ceilings.")
    
    st.header(":blue[Resale Levy:]")
    st.write("For buyers who have previously purchased a subsidized flat from the HDB, there might be a resale levy imposed when they purchase another flat from the HDB resale market.")
    
    st.header(":blue[Grant Schemes:]")
    st.write("There are various housing grant schemes available to eligible buyers, such as the CPF Housing Grant, which provides financial assistance for the purchase of resale flats.")
    
    st.header(":blue[HDB Loan and Bank Loan:]")
    st.write("Buyers can choose to finance their flat purchase through an HDB loan or a bank loan. HDB loans are provided by the HDB, while bank loans are obtained from commercial banks.")
    
    st.header(":blue[Market Trends:]")
    st.write("The resale market is influenced by various factors such as economic conditions, interest rates, and government policies. Property prices in Singapore can fluctuate based on these factors.")
    
    st.header(":blue[Online Platforms:]")
    st.write("There are online platforms and portals where sellers can list their resale flats, and buyers can browse available options.")


# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :orange[Predicting Results based on Trained Models]")
    st.markdown("### :green[Predicting Resale Price (Regression Task) (Accuracy: 87%)]")
    
    
    with st.form("form1"):
    
    # -----New Data inputs from the user for predicting the resale price-----
        st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
        street_name = st.text_input("Street Name")
        block = st.text_input("Block Number")
        floor_area_sqm = st.number_input('Floor Area (Per Square Meter) (min_value=30.0, max_value=300.0)', )
        lease_commence_year = st.number_input('Lease Commence Year (min_value = 1966, max_value=2024)')
        storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

        # -----Submit Button for PREDICT RESALE PRICE-----

        submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

        if submit_button:
            try: 
                loaded_model, scaler_loaded  = load_pickles()
                storey_median,lease_remain_years, min_dist_mrt, cbd_dist = preprocessing_input(street_name,block,lease_commence_year,storey_range)

                # -----Sending the user enter values for prediction to our model-----
                new_sample = np.array(
                    [[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                new_sample = scaler_loaded.transform(new_sample[:, :5])
                new_pred = loaded_model.predict(new_sample)[0]
                st.write('## :green[Predicted resale price:] ', np.floor(np.exp(new_pred)),':green[$]' )
            except Exception as err:
                st.warning("Please fill the form with Valid Details")