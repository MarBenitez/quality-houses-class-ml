import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import gzip
import shutil

# Configuración de la página de Streamlit
st.set_page_config(page_title="Swiss Houses Analysis and ML", layout="wide", page_icon=":house:")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .banner {
        background-image: url('https://uceap.universityofcalifornia.edu/sites/default/files/marketing-images/country-page-images/switzerland-page-header.jpg'); 
        background-size: cover;
        background-position: center;
        padding: 50px 0;
        text-align: center;
    }
    .title {
        font-family: 'Calibri', sans-serif;
        font-weight: 900; /* Increased font weight for bolder text */
        color: #66CDAA;
        text-align: center;
        font-size: 6em;
        margin-bottom: 0.5em;
        text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
    }
    .header {
        font-family: 'Calibri', sans-serif;
        color: #008B8B;
        font-weight: bold;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .subheader {
        font-family: 'Calibri', sans-serif;
        color: #F08080;
        text-align: center;
        font-size: 2em;
        margin-bottom: 0.5em;
    }
    .subsubheader {
        font-family: 'Calibri', sans-serif;
        color: #E9967A;
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 0.5em;
    }
    .subsubheader_sidebar {
        font-family: 'Calibri', sans-serif;
        font-weight: bold;
        color: #778899;
        font-size: 1.5em;
        margin-bottom: 0.5em;
    }
    .centered-text {
        font-family: 'Calibri', sans-serif;
        text-align: center;
        font-size: 1.15em;
        margin-bottom: 0.5em;
    }
    .scroll-up-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #8FBC8F;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        z-index: 100;
    }
    .stTabs [role="tablist"] {
        justify-content: center;
    }
    .about-us {
        text-align: center;
        font-family: 'Calibri', sans-serif;
        margin: 20px;
    }
    .about-us img {
        width: 100%;
        max-width: 600px;
        height: auto;
        margin: 20px 0;
    }
    </style>

    <button onclick="scrollToTop()" class="scroll-up-button">Scroll to Top</button>

    <script>
    function scrollToTop() {
        window.scrollTo({top: 0, behavior: 'smooth'});
    }
    </script>
    """,
    unsafe_allow_html=True
)

# Banner with title
st.markdown(
    """
    <div class="banner">
        <h1 class="title">Swiss Houses Project</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar with sections
st.sidebar.title("Navigation")
sections = ["Introduction", "Exploratory Data Analysis", "Clustering", "Classification", "User Application"]
selected_section = st.sidebar.selectbox("Go to", sections)

# Introduction Section
if selected_section == "Introduction":
    st.markdown('<div class="header">Introduction</div>', unsafe_allow_html=True)
    st.markdown(
    """
    <div class="about-us">
        <p>In this project, we aim to analyze a dataset of houses in Switzerland to uncover valuable insights and build <u><i>predictive models</u></i> for better understanding and predicting <b>house quality</b> in Switzerland.</p>
        <div style="display: flex; justify-content: center;">
            <img src="https://dynastytravel.com.sg/wp-content/uploads/2023/06/LUCERNE-1_switzerland_Lucerne-from-the-top-Switzerland.jpeg" alt="Switz" style="width: 45%; margin-right: 10px;">
            <img src="https://i.insider.com/5898d0f58275e829008b4842?width=1136&format=jpeg" alt="Mountains-switz" style="width: 45%;">
        </div>
    </div>
    """, unsafe_allow_html=True
    )
    st.markdown(
    """
    <div class="subheader"><b>Swiss Dwellings</b> Dataset</div>
    <div class="about-us">
        <p>The dataset initially contained <b>367 columns</b> of various features related to <i>house characteristics and environmental factors</i>. Due to the large number of features, we <u>selected the most valuable ones</u> for our analysis. Some of these features are:</p>
        <div style="display: flex; justify-content: center;">
        <table style="width: 80%; margin: auto; border-collapse: collapse;">
            <tr>
                <th style="border: 1px solid black; padding: 8px;">Feature</th>
                <th style="border: 1px solid black; padding: 8px;">Description</th>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">Window noise traffic night</td>
                <td style="border: 1px solid black; padding: 8px;">Noise level from traffic at night</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">Window noise train night</td>
                <td style="border: 1px solid black; padding: 8px;">Noise level from trains at night</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">View of buildings</td>
                <td style="border: 1px solid black; padding: 8px;">Amount of buildings views</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">View of greenery</td>
                <td style="border: 1px solid black; padding: 8px;">Amount of greenery views</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">View of ground</td>
                <td style="border: 1px solid black; padding: 8px;">Amount of ground views</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">View of pedestrians</td>
                <td style="border: 1px solid black; padding: 8px;">Amount of pedestrians views</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">View of street</td>
                <td style="border: 1px solid black; padding: 8px;">Amount of street views</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">Sunlight in the morning</td>
                <td style="border: 1px solid black; padding: 8px;">Amount of sunlight in the morning</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">Sunlight in the evening</td>
                <td style="border: 1px solid black; padding: 8px;">Amount of sunlight in the evening</td>
            </tr>
        </table>
        </div>
    </div>
    """, unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="subheader"><b>Project Sections</b></div>
        <div class="about-us">
            <p>The project is divided into <u>several sections</u> to explore, analyze, and model the dataset:</p>
            - Exploratory Data Analysis<br>
            - Clustering<br>
            - Classification<br>
            - User Application
        </div>
        """, unsafe_allow_html=True
    )




# Exploratory Data Analysis Section
if selected_section == "Exploratory Data Analysis":
    st.markdown('<div class="header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(r'<div class="centered-text">In this section, we explore the dataset and visualize various features.</div>', unsafe_allow_html=True)

    # Create tabs for different EDA sections
    tabs = st.tabs(["Overview", "Distribution Plots", "Correlation Heatmap", "Feature Analysis"])
    
    with tabs[0]:
        st.markdown('<div class="subheader">Overview</div>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">General overview and summary statistics of the dataset.</div>', unsafe_allow_html=True)
        # Display dataset overview
        df = pd.read_csv('data/df_feat_red.csv')
        st.markdown(r'<div class="subsubheader">General overview</div>', unsafe_allow_html=True)
        st.write(df.head())
        st.markdown(r'<div class="subsubheader">Summary statistics</div>', unsafe_allow_html=True)
        colT1,colT2= st.columns([26, 74])
        with colT2:
            st.write(df.describe().T)
        

    with tabs[1]:
        st.markdown('<div class="subheader"><b>Distribution Plots</b></div>', unsafe_allow_html=True)
        st.markdown(r'<div class="subsubheader"><b>Distribution of selected features</b></div>', unsafe_allow_html=True)
        
        # Plot distribution of selected features
        features = ['window_noise_traffic_night', 'window_noise_train_night', 'view_buildings_mean',
                    'view_greenery_mean', 'view_ground_mean', 'view_pedestrians_mean', 
                    'view_street_sum', 'mean_morning', 'mean_evening']
        
        # Create columns
        cols = st.columns(2)
        
        for i, feature in enumerate(features):
            # Format the feature name for title and x-label
            formatted_feature = feature.replace('_', ' ').title()
            
            # Select the appropriate column
            col = cols[i % 2]
            
            with col:
                fig, ax = plt.subplots(figsize=(5, 4))  # Smaller figure size
                sns.histplot(df[feature], bins=20, kde=True, ax=ax)
                ax.set_xlabel(formatted_feature)
                ax.set_title(f'Distribution of {formatted_feature}')
                st.pyplot(fig)



    with tabs[2]:
        st.markdown('<div class="subheader"><b>Correlation Heatmap</b></div>', unsafe_allow_html=True)
        st.markdown(r'<div class="subsubheader"><b>Heatmap of feature correlations</b></div>', unsafe_allow_html=True)
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation_matrix = df.select_dtypes(include='number').corr(method='spearman')
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        correlation_matrix_masked = correlation_matrix.mask(mask)
        sns.heatmap(correlation_matrix_masked, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        colT1,colT2,colT3 = st.columns([10,80,10])
        with colT2:
            st.pyplot(fig)

    with tabs[3]:
        st.markdown('<div class="subheader"><b>Feature Analysis</b></div>', unsafe_allow_html=True)
        st.markdown(r'<div class="subsubheader"><b>Detailed analysis of individual features</b></div>', unsafe_allow_html=True)
        
        # Feature analysis
        features = ['window_noise_traffic_night', 'window_noise_train_night', 'view_buildings_mean',
                    'view_greenery_mean', 'view_ground_mean', 'view_pedestrians_mean', 
                    'view_street_sum', 'mean_morning', 'mean_evening']
                    
        # Format feature names for selectbox
        formatted_features = [feature.replace('_', ' ').title() for feature in features]
        feature_map = dict(zip(formatted_features, features))
        
        selected_formatted_feature = st.selectbox("Select a feature for analysis", formatted_features)
        selected_feature = feature_map[selected_formatted_feature]
        
        st.markdown(f'<div class="centered-text"><b>Analysis of {selected_formatted_feature}</b></div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[selected_feature], ax=ax)
        ax.set_title(f'Boxplot of {selected_formatted_feature}')
        ax.set_xlabel(selected_formatted_feature)
        colT1,colT2,colT3 = st.columns([20,60,20])
        with colT2:
            st.pyplot(fig)
        
        st.markdown(f'<div class="centered-text"><b>Description of {selected_formatted_feature}</b></div>', unsafe_allow_html=True)
        
        # Center the table
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <table style="border-collapse: collapse;">
                    <tr>
                        <th style="border: 1px solid black; padding: 8px;">Statistic</th>
                        <th style="border: 1px solid black; padding: 8px;">Value</th>
                    </tr>
                    {''.join([f'<tr><td style="border: 1px solid black; padding: 8px;">{index}</td><td style="border: 1px solid black; padding: 8px;">{value}</td></tr>' for index, value in df[selected_feature].describe().items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )


# Clustering Section
elif selected_section == "Clustering":
    st.markdown('<div class="header">Clustering</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="centered-text">
            <p>In this section, we delve into the clustering analysis of the Swiss Houses dataset. Clustering is an unsupervised learning technique that helps us group similar data points based on their features. By applying clustering algorithms, we aim to identify distinct segments within the data, which can reveal underlying patterns and structures.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="centered-text">
            <p>- <b>DBSCAN</b>: Initially used, but resulted in imbalanced clusters.</p>
            <p>- <b>K-Means</b>: Selected for its ability to provide well-distributed clusters.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
        
    st.markdown('<div class="subheader"><b>Results</b></div>', unsafe_allow_html=True)
        
    # Create two columns for DBSCAN and K-Means results
    col1, col2 = st.columns(2)
        
    with col1:
        st.markdown('<div class="subsubheader centered-text"><b>DBSCAN</b></div>', unsafe_allow_html=True)
        st.write('<div class="centered-text">DBSCAN resulted in imbalanced clusters:</div>', unsafe_allow_html=True)
        # DBSCAN clusters table
        dbscan_data = {
            'Cluster': [0, -1, 4, 3, 1, 6, 5, 8, 7, 2, 10, 11, 9],
            'Count': [33517, 8438, 42, 32, 30, 29, 26, 20, 17, 16, 15, 15, 10]
        }
        dbscan_df = pd.DataFrame(dbscan_data).reset_index(drop=True)
        st.table(dbscan_df)
        
    with col2:
        st.markdown('<div class="subsubheader centered-text"><b>K-Means</b></div>', unsafe_allow_html=True)
        st.write('<div class="centered-text">K-Means resulted in well-balanced clusters:</div>', unsafe_allow_html=True)
        # K-Means clusters table
        kmeans_data = {
            'Cluster': [5, 1, 2, 4, 6, 0, 3, 7],
            'Count': [13651, 7361, 5396, 5188, 4353, 3914, 1853, 491]
        }
        kmeans_df = pd.DataFrame(kmeans_data).reset_index(drop=True)
        st.table(kmeans_df)
            
    st.write('<div class="centered-text">The final clustering results using K-Means with 8 clusters provided well-balanced groups.</div>', unsafe_allow_html=True)

    colT1,colT2,colT3 = st.columns([15,70,15])
    with colT2:
        st.image('images/clusters_8kmeans.png', use_column_width=True)

    colT1,colT2,colT3 = st.columns([15,70,15])
    with colT2:
        st.image('images/UMAP_8kmeans.png', use_column_width=True)


# Classification Section
elif selected_section == "Classification":
    st.markdown('<div class="header">Classification</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="centered-text">
            <p>In this section, we focus on building a supervised machine learning model to classify houses into different quality groups. We have developed and evaluated multiple classification models to identify the best-performing model for our dataset. Our ultimate goal is to create a reliable model that can accurately predict the quality of houses based on their features.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create tabs for Classification section
    classification_tabs = st.tabs(["Data", "Procedure", "Results"])



    with classification_tabs[0]:
        st.markdown('<div class="subheader"><b>Data</b></div>', unsafe_allow_html=True)
        st.markdown(
        """
        <div class="centered-text">
            <p>We discretized the variables before introducing them into the models in order to make the final application more user friendly.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
        # Load the data and model results (replace with actual data loading)
        df_results = pd.read_csv('data/df_class_disc6.csv')

        st.markdown('<div class="subsubheader centered-text"><b>Data discretized overview</b></div>', unsafe_allow_html=True)
        st.table(df_results.drop('apartment_id', axis=1).head())

    with classification_tabs[1]:
        st.markdown('<div class="subheader"><b>Procedure</b></div>', unsafe_allow_html=True)
        st.markdown(
        """
        <div class="centered-text">
            <p>We tested several classification algorithms and tuned their hyperparameters to achieve the best performance. The models we evaluated include:</p>
        </div>
        """,
        unsafe_allow_html=True
    )
        st.markdown(
        """
        <div class="centered-text">
            <p>- <b>Logistic Regression</b></p>
            <p>- <b>Random Forest</b></p>
            <p>- <b>K-Nearest Neighbours</b></p>
            <p>- <b>Support Vector Machine</b></p>
            <p>- <b>LightGBM</b></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

        st.markdown(
        """
        <div class="centered-text">
            <p>We also experimented with continuous and discretized features. The continuous features resulted in higher accuracy, but we chose to use discretized features for better interpretability and user engagement in our final application.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
        
        st.markdown(
    """
    <div class="about-us">
        <p>We also tried modelling with <b>Pycaret</b> and <b>AzureML</b>.</p>
        </div>
    </div>
    """, unsafe_allow_html=True
    )
        col1, col2 = st.columns(2)

        with col1:
            st.image('images/pycaret_results.png', caption='PyCaret Results', use_column_width=True, width=500)

        with col2:
            st.image('images/azure1.png', caption='Azure Results', use_column_width=True)

        
    with classification_tabs[2]:
        st.markdown('<div class="subheader"><b>Results</b></div>', unsafe_allow_html=True)

        st.markdown(
        """
        <div class="about-us">
            <p>Below are the results of the best-performing classification model using discretized features. The Random Forest classifier provided a good balance between accuracy and interpretability.</p>
            </div>
        </div>
        """, unsafe_allow_html=True
        )


        st.markdown('<div class="subsubheader centered-text"><b>Random Forest Classification Report</b></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <table style="border-collapse: collapse; width: 50%;">
                    <tr>
                        <th style="border: 1px solid black; padding: 8px;">Class</th>
                        <th style="border: 1px solid black; padding: 8px;">Precision</th>
                        <th style="border: 1px solid black; padding: 8px;">Recall</th>
                        <th style="border: 1px solid black; padding: 8px;">F1-Score</th>
                        <th style="border: 1px solid black; padding: 8px;">Support</th>
                    </tr>
                    <tr>
                        <td style="border: 1px solid black; padding: 8px;">1</td>
                        <td style="border: 1px solid black; padding: 8px;">0.63</td>
                        <td style="border: 1px solid black; padding: 8px;">0.62</td>
                        <td style="border: 1px solid black; padding: 8px;">0.63</td>
                        <td style="border: 1px solid black; padding: 8px;">1613</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid black; padding: 8px;">2</td>
                        <td style="border: 1px solid black; padding: 8px;">0.69</td>
                        <td style="border: 1px solid black; padding: 8px;">0.70</td>
                        <td style="border: 1px solid black; padding: 8px;">0.69</td>
                        <td style="border: 1px solid black; padding: 8px;">1446</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid black; padding: 8px;">3</td>
                        <td style="border: 1px solid black; padding: 8px;">0.67</td>
                        <td style="border: 1px solid black; padding: 8px;">0.78</td>
                        <td style="border: 1px solid black; padding: 8px;">0.72</td>
                        <td style="border: 1px solid black; padding: 8px;">1737</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid black; padding: 8px;">4</td>
                        <td style="border: 1px solid black; padding: 8px;">0.67</td>
                        <td style="border: 1px solid black; padding: 8px;">0.66</td>
                        <td style="border: 1px solid black; padding: 8px;">0.66</td>
                        <td style="border: 1px solid black; padding: 8px;">2226</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid black; padding: 8px;">5</td>
                        <td style="border: 1px solid black; padding: 8px;">0.66</td>
                        <td style="border: 1px solid black; padding: 8px;">0.63</td>
                        <td style="border: 1px solid black; padding: 8px;">0.64</td>
                        <td style="border: 1px solid black; padding: 8px;">4074</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid black; padding: 8px;">6</td>
                        <td style="border: 1px solid black; padding: 8px;">0.74</td>
                        <td style="border: 1px solid black; padding: 8px;">0.72</td>
                        <td style="border: 1px solid black; padding: 8px;">0.73</td>
                        <td style="border: 1px solid black; padding: 8px;">1567</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid black; padding: 8px;" colspan="4">Accuracy</td>
                        <td style="border: 1px solid black; padding: 8px;">0.67</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid black; padding: 8px;" colspan="4">Macro avg</td>
                        <td style="border: 1px solid black; padding: 8px;">0.68</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid black; padding: 8px;" colspan="4">Weighted avg</td>
                        <td style="border: 1px solid black; padding: 8px;">0.67</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.write('<div class="centered-text"><u>Model final Accuracy:</u> <b>0.6715628208165522</b></div>', unsafe_allow_html=True)


        colT1,colT2,colT3 = st.columns([20,60,20])
        with colT2:
            st.image('images/final_conf_matrix.png', use_column_width=True)


# User Application Section
elif selected_section == "User Application":
    st.markdown('<div class="header">User Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="centered-text"><u>Predict the group of your house by entering its characteristics:</u></div>', unsafe_allow_html=True)
    st.markdown('<div class="centered-text"><i>Note that the data are in terms of quantity, where 1 is a little and 4 is a lot.</i></div>', unsafe_allow_html=True)
    
    # Input features
    colT1, colT2, colT3 = st.columns([20, 60, 20])
    with colT2:
        window_noise_traffic_night = st.selectbox('Traffic Noise at Night', [1, 2, 3, 4])
        window_noise_train_night = st.selectbox('Train Noise at Night', [1, 2, 3, 4])
        view_buildings_mean = st.selectbox('View of Buildings', [1, 2, 3, 4])
        view_greenery_mean = st.selectbox('View of Greenery', [1, 2, 3, 4])
        view_ground_mean = st.selectbox('View of Ground', [1, 2, 3, 4])
        view_pedestrians_mean = st.selectbox('View of Pedestrians', [1, 2, 3, 4])
        view_street_sum = st.selectbox('View of Street', [1, 2, 3, 4])
        mean_morning = st.selectbox('Sunlight in the Morning', [1, 2, 3, 4])
        mean_evening = st.selectbox('Sunlight in the Evening', [1, 2, 3, 4])

    # Create DataFrame for prediction
    input_data = pd.DataFrame([[
        window_noise_traffic_night, window_noise_train_night, view_buildings_mean,
        view_greenery_mean, view_ground_mean, view_pedestrians_mean,
        view_street_sum, mean_morning, mean_evening
    ]], columns=[
        'window_noise_traffic_night', 'window_noise_train_night', 'view_buildings_mean',
        'view_greenery_mean', 'view_ground_mean', 'view_pedestrians_mean',
        'view_street_sum', 'mean_morning', 'mean_evening'
    ])

    # Function to decompress and load model
    def load_compressed_model(file_path):
        with gzip.open(file_path, 'rb') as f_in:
            with open('decompressed_model.pkl', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        model = joblib.load('decompressed_model.pkl')
        return model

    # Load the model
    model = load_compressed_model('rf_model_8C_disc.pkl.gz')

    # model1: rf_model_6CLUSTERS_disc_num.pkl.gz
    # model2: rf_model_6C_disc.pkl.gz
    # model3: rf_model_8C_disc.pkl.gz

    colT1, colT2, colT3 = st.columns([20, 60, 20])
    with colT2:
        if st.button('Predict'):
            prediction = model.predict(input_data)
            st.write(f"Your house belongs to cluster: {prediction[0]}")

    # Interpretation of Results
    st.markdown('<div class="subheader">Interpretation of Results</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <table style="border-collapse: collapse; width: 80%;">
                <tr>
                    <th style="border: 1px solid black; padding: 8px;">Cluster</th>
                    <th style="border: 1px solid black; padding: 8px;">Description</th>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">1</td>
                    <td style="border: 1px solid black; padding: 8px;">This represents the houses of the lowest quality. These houses might have higher levels of noise, less greenery, and poorer sunlight exposure.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">2</td>
                    <td style="border: 1px solid black; padding: 8px;">These houses are slightly better than those in Cluster 1 but still fall on the lower end of the quality spectrum.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">3</td>
                    <td style="border: 1px solid black; padding: 8px;">Houses in this cluster are of moderate quality, with balanced features that provide a reasonable living environment.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">4</td>
                    <td style="border: 1px solid black; padding: 8px;">These houses have good quality, with better views and noise conditions compared to the lower clusters.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">5</td>
                    <td style="border: 1px solid black; padding: 8px;">This cluster includes houses that are of high quality, offering better views, less noise, and more greenery.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">6</td>
                    <td style="border: 1px solid black; padding: 8px;">Representing the houses of the highest quality, these houses have the best views, the least noise, and the most sunlight exposure.</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

# End of script
