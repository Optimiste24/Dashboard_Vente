# vente.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Configuration initiale
def configure_visuals():
    sns.set_style("whitegrid")
    plt.rc('figure', autolayout=True, figsize=(11, 4))
    plt.rc('axes', labelweight='bold', labelsize='large',
           titleweight='bold', titlesize=16, titlepad=10)

# Configuration de la page
st.set_page_config(
    page_title="Favorita Sales Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Chargement des données depuis GCS
@st.cache_data
def load_data():
    base_url = "https://storage.googleapis.com/venteequateur/data/"
    
    train_files = [
        "train_2013.csv", "train_2014.csv",
        "train_2015.csv", "train_2016.csv", "train_2017.csv"
    ]
    
    # Chargement des fichiers d'entraînement
    train_dfs = []
    for file in train_files:
        try:
            df = pd.read_csv(base_url + file, parse_dates=['date'])
            train_dfs.append(df)
        except Exception as e:
            st.error(f"Erreur chargement {file}: {str(e)}")
    
    data = {
        'train': pd.concat(train_dfs, ignore_index=True) if train_dfs else None,
        'stores': pd.read_csv(base_url + 'stores.csv'),
        'oil': pd.read_csv(base_url + 'oil.csv', parse_dates=['date']),
        'holidays': pd.read_csv(base_url + 'holidays_events.csv', parse_dates=['date'])
    }
    return data


def about_page():
    st.title("👨‍💻 À Propos de Moi")
    
    # Section Profil avec colonnes
    col1, col2 = st.columns([0.3, 0.7])
    
    with col1:
        # Lien direct vers l'image (remplacé par un hébergement public valide)
        st.image("https://storage.googleapis.com/venteequateur/data/Profil_pro.JPG", 
                width=200,
                caption="Yassh Agoro")
    
    with col2:
        st.markdown("""
        **Yassh Agoro**  
        *Data Scientist Consultant | Expert en Analyse Prédictive*  
        """)
        
        # Badges interactifs
        st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/yassh-agoro-91a460315")
        st.link_button("🌟 Malt", "https://www.malt.fr/profile/yasshagoro1")
    
    # Valeur proposition
    st.header("🎯 Ma Mission")
    st.markdown("""
    > *Je transforme les données des commerces, franchises et PME en **décisions stratégiques et actions concrètes**.*  
    > *Combinaison unique d'expertise terrain (vente/gestion) et technique (data science/automatisation).*
    """)
    
    # Services sous forme de cartes
    st.header("💡 Mes Services")
    
    services = [
        ("📊", "**Suivi temps réel des KPI**", "Streamlit, Power BI, Excel VBA"),
        ("⚙️", "**Automatisation des reportings**", "Gain de temps jusqu'à 80%"),
        ("🔮", "**Prédiction des ventes**", "Modèles SARIMAX, XGBoost, LSTM"),
        ("📈", "**Optimisation rentabilité**", "Analyse produits/pricing/clients"),
        ("🖥️", "**Dashboards stratégiques**", "Visuels simples et actionnables")
    ]
    
    for icon, title, desc in services:
        with st.expander(f"{icon} {title}"):
            st.caption(desc)
    
    # Projet actuel
    st.header("🚀 Projet Actuel")
    st.markdown("""
    **📊 Prédiction des ventes quotidiennes par magasin/famille**  
    *Pour une chaîne de distribution équatorienne (Dataset: Corporación Favorita)*  
    
    - **Objectif** : Anticiper les demandes avec <90% de précision (RMSE)  
    - **Technos** : Python, Streamlit, LightGBM, Prophet  
    - **Livrables** : Dashboard interactif + API de prédiction  
    """)
    
    # Compétences techniques (avec liens d'images valides)
    st.header("🛠 Stack Technique")
    
    techs = {
        "Python": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png",
        "Streamlit": "https://streamlit.io/images/brand/streamlit-mark-color.png",
        "Power BI": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Power_bi_logo_black.svg/1200px-Power_bi_logo_black.svg.png",
        "SQL": "https://cdn-icons-png.flaticon.com/512/4492/4492311.png"
    }
    
    cols = st.columns(len(techs))
    for col, (name, url) in zip(cols, techs.items()):
        with col:
            st.image(url, width=60)
            st.caption(f"**{name}**")
    
    # Contact
    st.divider()
    st.markdown("""
    ✉️ **Contact** : [yagoropro@outlook.fr](mailto:yagoropro@outlook.fr)  
    📞 **Téléphone** : +33 9 51 79 59 24  
    """)
def eda_page(data):
    st.title("📊 Exploration des Données (EDA)")
    st.markdown("""
    **Objectif:** Identifier les tendances, saisonnalités et relations clés dans les données.
    """)

    # Vérification que les données sont chargées
    if data is None or 'train' not in data:
        st.error("Erreur: Les données n'ont pas pu être chargées!")
        return

    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Ventes", "🏪 Magasins", "⛽ Pétrole", "🎉 Jours Fériés"])

    # 1. Analyse des Ventes
    with tab1:
        st.header("Analyse des Ventes Temporelles")
        
        # Convertir la date si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(data['train']['date']):
            data['train']['date'] = pd.to_datetime(data['train']['date'])
        
        # Sélecteur de période
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Date début", data['train']['date'].min())
        with col2:
            end_date = st.date_input("Date fin", data['train']['date'].max())
        
        # Filtrage des données
        filtered = data['train'][
            (data['train']['date'] >= pd.to_datetime(start_date)) & 
            (data['train']['date'] <= pd.to_datetime(end_date))
        ]
        
        # Agrégation interactive
        freq = st.radio("Fréquence", ["Journalier", "Hebdomadaire", "Mensuel"], horizontal=True)
        
        if freq == "Journalier":
            sales = filtered.groupby('date')['sales'].sum()
        elif freq == "Hebdomadaire":
            sales = filtered.set_index('date').resample('W-Mon')['sales'].sum()
        else:
            sales = filtered.set_index('date').resample('ME')['sales'].sum()
        
        # Visualisation avec Plotly
        fig = px.line(sales, title=f"Ventes {freq.lower()}s")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Analyse Géographique des Magasins")
        
        # Fusion avec les données magasins
        merged = pd.merge(data['train'], data['stores'], on='store_nbr')
        
        # Sélection du type d'analyse
        analysis_type = st.selectbox("Vue par:", ["Ville", "Type de Magasin"])
        
        if analysis_type == "Ville":
            fig = px.bar(merged.groupby('city')['sales'].sum().sort_values(ascending=False),
                         title="Ventes Totales par Ville")
        elif analysis_type == "Type de Magasin":
            fig = px.pie(merged.groupby('type')['sales'].sum(), 
                         names=merged['type'].unique(), 
                         title="Répartition des Ventes par Type de Magasin")
#        else:
#            fig = px.box(merged, x='cluster', y='sales', 
#                         title="Distribution des Ventes par Cluster")
        
        st.plotly_chart(fig, use_container_width=True)


    # 2. Analyse Pétrole-Ventes (avec corrélation colorée)
    with tab3:
        st.header("Analyse Prix du Pétrole vs Ventes")
        
        # Fusion des données
        oil_sales = pd.merge(
            data['train'].groupby('date')['sales'].sum().reset_index(),
            data['oil'],
            on='date',
            how='inner'
        ).dropna()
        
        # Calcul de corrélation
        corr = oil_sales['sales'].corr(oil_sales['dcoilwtico'])
        st.metric("Corrélation globale", f"{corr:.2f}")
        
        # Graphique avec double axe et gradient de couleur
        fig = go.Figure()
        
        # Ajout des ventes (axe Y gauche)
        fig.add_trace(go.Scatter(
            x=oil_sales['date'],
            y=oil_sales['sales'],
            name="Ventes",
            line=dict(color='royalblue')
        ))
        
        # Ajout du pétrole (axe Y droit)
        fig.add_trace(go.Scatter(
            x=oil_sales['date'],
            y=oil_sales['dcoilwtico'],
            name="Prix du Pétrole",
            yaxis="y2",
            line=dict(color='crimson')
        ))
        
        # Mise en forme
        fig.update_layout(
            title="Relation Temporelle: Ventes vs Prix du Pétrole",
            yaxis=dict(title="Ventes", side="left"),
            yaxis2=dict(title="Prix du Pétrole", overlaying="y", side="right"),
            hovermode="x unified",
            colorway=["#636EFA", "#EF553B"]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap de corrélation par mois
        oil_sales['month'] = oil_sales['date'].dt.month
        monthly_corr = oil_sales.groupby('month')[['sales', 'dcoilwtico']].corr().iloc[0::2,-1].reset_index()
        
        fig = px.bar(monthly_corr, x='month', y='dcoilwtico', 
                     color='dcoilwtico',
                     color_continuous_scale=px.colors.diverging.RdBu,
                     range_color=[-1, 1],
                     title="Corrélation par Mois")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Impact des Jours Fériés")
        # Boxplot ventes vs fériés
        holidays = data['holidays'][data['holidays']['locale'] == 'National']
        merged = pd.merge(data['train'], holidays[['date', 'type']], on='date', how='left')
        merged['is_holiday'] = merged['type'].notna()
        st.bar_chart(merged.groupby('is_holiday')['sales'].mean())

def feature_engineering_page(data):
    st.title("🛠 Feature Engineering")
    
    with st.expander("⏱ Features Temporelles", expanded=True):
        # Exemple de création de features
        if st.button("Ajouter les caractéristiques temporelles"):
            data['train']['day_of_week'] = data['train']['date'].dt.dayofweek
            data['train']['month'] = data['train']['date'].dt.month
            st.success("Features temporelles ajoutées !")
    
    with st.expander("🏪 Features Magasins"):
        if st.button("Ajouter les clusters de magasin"):
            data['train'] = pd.merge(data['train'], data['stores'][['store_nbr', 'cluster']], on='store_nbr')
            st.success("Clusters ajoutés !")
    
    with st.expander("📅 Jours Spéciaux"):
        # Exemple pour le tremblement de terre
        if st.button("Marquer la période post-tremblement de terre"):
            earthquake_date = pd.to_datetime('2016-04-16')
            data['train']['post_earthquake'] = (
                (data['train']['date'] >= earthquake_date) & 
                (data['train']['date'] <= earthquake_date + pd.Timedelta(days=30))
            )
            st.success("Période marquée !")

def preprocessing_page():
    st.title("🧹 Prétraitement")
    st.warning("Section en construction - Disponible prochainement")
    # Placeholder pour les fonctions de prétraitement

def modeling_page():
    st.title("🤖 Modélisation")
    st.warning("Section en construction - Disponible prochainement")
    # Placeholder pour les fonctions de modélisation

def evaluation_page():
    st.title("📈 Post-traitement & Évaluation")
    st.warning("Section en construction - Disponible prochainement")
    # Placeholder pour les fonctions d'évaluation

# Navigation principale
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", [
        "À Propos de Moi",
        "Exploration des Données (EDA)",
        "Feature Engineering",
        "Prétraitement",
        "Modélisation",
        "Post-traitement & Évaluation"
    ])
    
    # Chargement des données (uniquement pour les pages qui en ont besoin)
    data = None
    if page in ["Exploration des Données (EDA)", "Feature Engineering"]:
        data = load_data()
    
    # Router vers la page sélectionnée
    if page == "À Propos de Moi":
        about_page()
    elif page == "Exploration des Données (EDA)":
        eda_page(data)
    elif page == "Feature Engineering":
        feature_engineering_page(data)
    elif page == "Prétraitement":
        preprocessing_page()
    elif page == "Modélisation":
        modeling_page()
    elif page == "Post-traitement & Évaluation":
        evaluation_page()

if __name__ == "__main__":
    main()
