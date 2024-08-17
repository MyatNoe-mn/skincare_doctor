
import streamlit as st
from streamlit_option_menu import option_menu
from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Import the Dataset
file_path = r"C:\VScode\MP-Skin Care Product Recommendation System.csv"
skincare = pd.read_csv(file_path, encoding='utf-8', index_col=None)

# Set the header for the Streamlit app
st.set_page_config(page_title="Skin Care Recommender System", page_icon=":rose:", layout="wide")

# Example number for menu style
EXAMPLE_NO = 2

def streamlit_menu(example=1):
    if example == 1:
        # Sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                options=["Skin Care", "Get Recommendation", "Skin Care 101"],
                icons=["house", "stars", "book"],
                menu_icon="cast",
                default_index=0,
            )
        return selected

    if example == 2:
        # Horizontal menu without custom style
        selected = option_menu(
            menu_title=None,
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],
            icons=["house", "stars", "book"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # Horizontal menu with custom style
        selected = option_menu(
            menu_title=None,
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],
            icons=["house", "stars", "book"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Skin Care":
    st.title(f"{selected} Product Recommender :sparkles:")
    st.write('---') 

    st.write(
        """
        ##### **This Skin Care Product Recommendation application uses Machine Learning to suggest skincare products based on your skin type and issues.**
        """)
    
   
    st.write(' ') 
    st.write(' ')
    st.write(
        """
        ##### You will receive skincare product recommendations from various cosmetic brands with over 1200 products tailored to your skin needs. 
        ##### There are 5 categories of skincare products for 5 different skin types, as well as issues and benefits. This recommendation system provides suggestions based on the data you enter, not scientific consultation.
        ##### Please select the *Get Recommendation* page to start receiving recommendations or choose the *Skin Care 101* page to see skincare tips and tricks.
        """)
    
    st.write(
        """
        **Happy Trying :) !**
        """)
    
    st.info('Credit: Created by Dwi Ayu Nouvalina')

if selected == "Get Recommendation":
    st.title(f"Let's {selected}")
    
    st.write(
        """
        ##### **To get recommendations, please enter your skin type, issues, and desired benefits to receive the right skincare product recommendations.**
        """) 
    
    st.write('---') 

    first, last = st.columns(2)

    # Choose a product category
    category = first.selectbox(label='Product Category: ', options=skincare['product_type'].unique())
    category_pt = skincare[skincare['product_type'] == category]
    print(category_pt.columns)

    # Choose a skin type
    skin_type = last.selectbox(label='Your Skin Type: ', options=['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'])
    category_st_pt = category_pt[category_pt['skintype'] == skin_type]

    # Select skin problems
    prob = st.multiselect(label='Skin Problems: ', options=['Dull Skin', 'Acne', 'Acne Scars', 'Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin'])
    

    # Choose notable effects
    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    selected_options = st.multiselect('Notable Effects: ', opsi_ne)
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    # Choose product
    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    product = st.selectbox(label='Recommended Product for You', options=sorted(opsi_pn))

    # MODELLING with Content Based Filtering
    tf = TfidfVectorizer()

    # Calculate idf for 'notable_effects'
    tf.fit(skincare['notable_effects']) 

    # Map feature index to feature names
    tf.get_feature_names_out()

    # Transform data to matrix
    tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 

    # Check tfidf matrix size
    shape = tfidf_matrix.shape

    # Convert tf-idf matrix to dense format
    tfidf_matrix.todense()

    # Create dataframe to view tf-idf matrix
    pd.DataFrame(
        tfidf_matrix.todense(), 
        columns=tf.get_feature_names_out(),
        index=skincare.product_name
    ).sample(shape[1], axis=1).sample(10, axis=0)

    # Compute cosine similarity on tf-idf matrix
    cosine_sim = cosine_similarity(tfidf_matrix) 

    # Create dataframe for cosine similarity
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    # View similarity matrix
    cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

    # Function to get recommendations
    def skincare_recommendations(product_name, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description']], k=5):

        # Find the most similar products
        index = similarity_data.loc[:, product_name].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1:-(k+2):-1]]

        # Exclude the selected product from recommendations
        closest = closest.drop(product_name, errors='ignore')
        df = pd.DataFrame(closest).merge(items).head(k)
        return df

    # Button to display recommendations
    model_run = st.button('Find Other Similar Products!')
    if model_run:
        st.write('Here are other similar product recommendations based on your preferences:')
        st.write(skincare_recommendations(product))
    
if selected == "Skin Care 101":
    st.title(f"Take a Look at {selected}")
    st.write('---') 

    st.write(
        """
        ##### **Here are some tips and tricks to maximize the use of skincare products**
        """) 
    
    
    st.write(
        """
        ### **1. Facial Wash**
        """)
    st.write(
        """
        **- Use a recommended facial wash product or one that suits you**
        """)
    st.write(
        """
        **- Wash your face up to twice a day, in the morning and before bed. Washing your face too often can strip away natural oils. For those with dry skin, it's fine to use just water in the morning.**
        """)
    st.write(
        """
        **- Avoid scrubbing your face harshly as it can remove the skin's natural barrier.**
        """)
    st.write(
        """
        **- The best way to clean your skin is to use your fingertips in a circular motion and massage for 30-60 seconds.**
        """)
    
    st.write(
        """
        ### **2. Toner**
        """)
    st.write(
        """
        **- Use a recommended toner or one that suits you.**
        """)
    st.write(
        """
        **- Apply toner to a cotton pad and gently swipe over your face. For best results, use two layers of toner: the first with a cotton pad and the last with your hands for better absorption.**
        """)
    st.write(
        """
        **- Use toner after cleansing your face.**
        """)
    st.write(
        """
        **- For sensitive skin, avoid skincare products containing fragrance as much as possible.**
        """)
    
    st.write(
        """
        ### **3. Serum**
        """)
    st.write(
        """
        **- Use a recommended serum or one that suits you for best results.**
        """)
    st.write(
        """
        **- Apply serum after your face is completely clean to ensure the serum absorbs fully.**
        """)
    st.write(
        """
        **- Use serum in the morning and at night before bed.**
        """)
    st.write(
        """
         **- Choose a serum according to your needs, such as for acne scars,**
         """)
    
    st.write(
        """
        ### 4. Moisturizer
        """)
    st.write(
        """
        
Use a recommended moisturizer or one that suits you for best results.

        """)
    st.write(
        """
        
Apply moisturizer after one or two minutes of using serum to allow the product to absorb into your skin.

        """)
    st.write(
        """
        
For better result, use moisturizer in the morning and at night before bed.

        """)
    st.write(
        """
         
The massaging affect that is used when applying moisturizer helps stimulate blood circulation and new cell generation,

         """)
    st.write(
        """
        ### 5. Sunscreen
        """)
    st.write(
        """
        
Use a recommended sunscreen or one that suits you for best results.

        """)
    st.write(
        """
        
A physical sunscreen should always be applied after your moisturizer.

        """)
    st.write(
        """
        
When it comes to applying sunscreen on your face, the recommended amount is around 1⁄4 teaspoon to get its protective effect.

        """)
    st.write(
        """
         
The active ingredients in sunscreens are oils that need to be removed from the skin before bedtime,

         """)
    
