import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Projet Scoring Crédit", layout="wide")

# ==========================================
# FONCTIONS DE CHARGEMENT
# ==========================================
@st.cache_data
def load_raw_data():
    path = 'data/Loan_Data.csv' if os.path.exists('data/Loan_Data.csv') else '../data/Loan_Data.csv'
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_ml_objects():
    base_path = 'data/processed/' if os.path.exists('data/processed/') else '../data/processed/'
    try:
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        model = joblib.load(os.path.join(base_path, 'best_model.pkl'))
        return scaler, model
    except FileNotFoundError:
        return None, None

df_raw = load_raw_data()
scaler, best_model = load_ml_objects()

# ==========================================
# MENU LATÉRAL
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", [
    "Accueil",
    "01 - Analyse Exploratoire",
    "02 - Pré-traitement",
    "03 - Modélisation & Simulateur"
])

# ==========================================
# PAGE : ACCUEIL
# ==========================================
if page == "Accueil":
    st.title("💳 Application de Scoring de Crédit")
    st.write("""
    Bienvenue dans notre tableau de bord de Data Science dédié à la prédiction du risque de défaut de paiement.
    
    Ce projet complet est divisé en trois grandes parties accessibles via le menu de gauche :
    * **01 - Analyse Exploratoire :** Visualisation et compréhension de nos 10 000 clients.
    * **02 - Pré-traitement :** Nettoyage, création de variables (Feature Engineering) et équilibrage des données (SMOTE).
    * **03 - Modélisation & Simulateur :** Comparaison de nos algorithmes de Machine Learning et outil de prédiction en direct.
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clients", "10 000")
    with col2:
        st.metric("Taux de Défaut", "18.5%")
    with col3:
        st.metric("Meilleur Modèle", "Random Forest")
    
    st.info("👈 Utilisez le menu latéral pour commencer la navigation.")

# ==========================================
# PAGE 1 : EDA
# ==========================================
elif page == "01 - Analyse Exploratoire":
    st.title("📊 Analyse Exploratoire des Données (EDA)")
    
    df_eda = df_raw.copy()
    if 'customer_id' in df_eda.columns:
        df_eda.set_index('customer_id', inplace=True)
    
    st.header("1. Aperçu des données")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_eda.head())
    with col2:
        st.write(f"**Total clients :** {len(df_eda):,}")
        st.write(f"**Taux de défaut global :** {df_eda['default'].mean()*100:.2f}%")
        st.write(f"**Valeurs manquantes :** {df_eda.isnull().sum().sum()}")
        st.write(f"**Doublons :** {df_eda.duplicated().sum()}")
    
    st.markdown("---")
    
    st.header("2. Distribution de la variable cible")
    fig_target, ax_target = plt.subplots(figsize=(5, 3))
    counts = df_eda['default'].value_counts()
    ax_target.bar(['Non Défaut (0)', 'Défaut (1)'], counts.values, 
                  color=['#2ecc71', '#e74c3c'], edgecolor='white')
    ax_target.set_title("Distribution des défauts de paiement")
    for i, v in enumerate(counts.values):
        ax_target.text(i, v + 50, f'{v/len(df_eda)*100:.1f}%', ha='center', fontsize=12)
    st.pyplot(fig_target)
    
    st.markdown("---")
    
    st.header("3. Distribution des variables")
    features = ['credit_lines_outstanding', 'loan_amt_outstanding', 
                'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    
    tab1, tab2 = st.tabs(["Histogrammes", "Boxplots (vs Défaut)"])
    
    with tab1:
        fig_hist, axes_hist = plt.subplots(2, 3, figsize=(15, 8))
        axes_hist = axes_hist.flatten()
        for i, col in enumerate(features):
            axes_hist[i].hist(df_eda[col], bins=40, color='steelblue', edgecolor='white')
            axes_hist[i].set_title(f'Distribution : {col}', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_hist)
        
    with tab2:
        fig_box, axes_box = plt.subplots(2, 3, figsize=(15, 8))
        axes_box = axes_box.flatten()
        for i, col in enumerate(features):
            sns.boxplot(x='default', y=col, data=df_eda, hue='default',
                       palette={0: '#2ecc71', 1: '#e74c3c'}, legend=False, ax=axes_box[i])
            axes_box[i].set_title(f'{col} vs Default', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_box)

    st.markdown("---")

    st.header("4. Focus sur le FICO Score")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_fico, ax_fico = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df_eda, x='fico_score', hue='default', bins=40, 
                    palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.7, ax=ax_fico)
        plt.title("FICO Score selon le statut de défaut", fontsize=14)
        st.pyplot(fig_fico)
    with col2:
        fico_default = df_eda[df_eda['default']==1]['fico_score'].mean()
        fico_no_default = df_eda[df_eda['default']==0]['fico_score'].mean()
        st.metric("FICO moyen (Non Défaut)", f"{fico_no_default:.0f}")
        st.metric("FICO moyen (Défaut)", f"{fico_default:.0f}")
        st.info("**Interprétation :** Les faibles scores FICO sont fortement corrélés au risque de défaut.")

    st.markdown("---")

    st.header("5. Matrice de Corrélation")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    corr = df_eda.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
               center=0, square=True, linewidths=0.5, ax=ax_corr)
    plt.title("Matrice de Corrélation", fontsize=14)
    st.pyplot(fig_corr)


# ==========================================
# PAGE 2 : PREPROCESSING
# ==========================================
elif page == "02 - Pré-traitement":
    st.title("⚙️ Pré-traitement (Feature Engineering & SMOTE)")
    
    st.header("1. Création de nouvelles variables (Feature Engineering)")
    st.write("Nous avons enrichi nos données en créant deux ratios financiers très utilisés dans le secteur bancaire :")
    
    col1, col2 = st.columns(2)
    with col1:
        st.code("""
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']
df['loan_to_income'] = df['loan_amt_outstanding'] / df['income']
        """, language="python")
    with col2:
        st.info("**debt_to_income** : Mesure le niveau d'endettement global par rapport au revenu.\n\n**loan_to_income** : Mesure le poids du prêt demandé par rapport au revenu.")
    
    st.markdown("---")
    
    st.header("2. Mise à l'échelle (StandardScaler)")
    st.write("Toutes les variables sont ramenées à la même échelle (moyenne = 0, écart-type = 1). Le scaler est sauvegardé pour les prédictions futures.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avant Scaling (income moyen)", "~66 000 $")
    with col2:
        st.metric("Après Scaling (income moyen)", "~0.00")
    with col3:
        st.metric("Fichier sauvegardé", "scaler.pkl ✅")
    
    st.markdown("---")
    
    st.header("3. Rééquilibrage avec SMOTE")
    st.write("La variable cible étant déséquilibrée (18.5% de défauts), SMOTE génère des données synthétiques sur le jeu d'entraînement.")
    
    fig_smote, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(['Non Défaut (0)', 'Défaut (1)'], [6519, 1481], 
            color=['#2ecc71', '#e74c3c'], edgecolor='white')
    ax1.set_title('Avant SMOTE (Train Set)')
    ax1.set_ylabel('Nombre de clients')
    for i, v in enumerate([6519, 1481]):
        ax1.text(i, v + 50, str(v), ha='center', fontsize=11)
    
    ax2.bar(['Non Défaut (0)', 'Défaut (1)'], [6519, 6519], 
            color=['#2ecc71', '#e74c3c'], edgecolor='white')
    ax2.set_title('Après SMOTE (Train Set)')
    ax2.set_ylabel('Nombre de clients')
    for i, v in enumerate([6519, 6519]):
        ax2.text(i, v + 50, str(v), ha='center', fontsize=11)
    
    plt.tight_layout()
    st.pyplot(fig_smote)
    st.success("✅ Le jeu d'entraînement est désormais parfaitement équilibré à 50% / 50% !")

    st.markdown("---")
    
    st.header("4. Résumé du Pipeline")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Features originales", "7")
    with col2:
        st.metric("Features après FE", "8")
    with col3:
        st.metric("X_train (après SMOTE)", "13 038")
    with col4:
        st.metric("X_test", "2 000")

# ==========================================
# PAGE 3 : MODELISATION & SIMULATEUR
# ==========================================
elif page == "03 - Modélisation & Simulateur":
    st.title("🤖 Modélisation ML & Simulateur de Crédit")
    
    st.header("1. Comparaison des algorithmes (MLflow)")
    st.write("Trois algorithmes ont été testés et trackés via MLflow :")
    
    results = {
        "Modèle": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.9970, 0.9965, 0.9930],
        "F1 Score": [0.9920, 0.9905, 0.9811],
        "ROC-AUC": [1.0000, 0.9937, 0.9998],
        "Precision": [0.9840, 0.9919, 0.9811],
        "Recall": [1.0000, 0.9892, 0.9811]
    }
    df_results = pd.DataFrame(results).set_index("Modèle")
    st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'))
    
    # Graphique comparaison
    fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
    df_results.plot(kind='bar', ax=ax_comp, colormap='Set2')
    ax_comp.set_title('Comparaison des métriques par modèle')
    ax_comp.set_ylim(0.95, 1.01)
    ax_comp.set_xticklabels(df_results.index, rotation=15)
    ax_comp.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig_comp)
    
    st.success("🏆 **Meilleur modèle retenu : Random Forest** (ROC-AUC = 0.9998)")
    
    st.markdown("---")
    
    st.header("🔮 Simulateur Interactif de Risque de Crédit")
    
    if scaler is None or best_model is None:
        st.error("⚠️ Modèle ou scaler introuvable. Vérifie que data/processed/best_model.pkl et scaler.pkl existent.")
    else:
        st.write("Entrez le profil d'un client pour évaluer son risque de défaut :")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Informations financières")
                income = st.number_input("Revenu annuel ($)", min_value=1000.0, value=70000.0, step=1000.0)
                loan_amt = st.number_input("Montant du prêt ($)", min_value=0.0, value=4000.0, step=500.0)
                total_debt = st.number_input("Dette totale en cours ($)", min_value=0.0, value=8000.0, step=500.0)
                
            with col2:
                st.subheader("👤 Profil du client")
                fico = st.slider("Score FICO", min_value=300, max_value=850, value=640)
                credit_lines = st.number_input("Lignes de crédit", min_value=0, value=1, step=1)
                years_employed = st.number_input("Années d'emploi", min_value=0, max_value=50, value=5, step=1)
                
            submit = st.form_submit_button("🚀 Lancer l'analyse de risque", use_container_width=True)
            
        if submit:
            # Feature Engineering
            debt_to_income = total_debt / income if income > 0 else 0
            loan_to_income = loan_amt / income if income > 0 else 0
            
            # Création du DataFrame
            input_features = ['credit_lines_outstanding', 'loan_amt_outstanding', 
                             'total_debt_outstanding', 'income', 'years_employed', 
                             'fico_score', 'debt_to_income', 'loan_to_income']
            
            user_data = pd.DataFrame([[credit_lines, loan_amt, total_debt, income, 
                                      years_employed, fico, debt_to_income, loan_to_income]], 
                                     columns=input_features)
            
            # Scaling et Prédiction
            user_data_scaled = scaler.transform(user_data)
            prediction = best_model.predict(user_data_scaled)[0]
            proba = best_model.predict_proba(user_data_scaled)[0][1]
            
            # Résultat
            st.markdown("---")
            st.markdown("### 📋 Résultat de l'analyse :")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probabilité de défaut", f"{proba:.2%}")
            with col2:
                st.metric("Ratio Dette/Revenu", f"{debt_to_income:.2%}")
            with col3:
                st.metric("Ratio Prêt/Revenu", f"{loan_to_income:.2%}")
            
            st.markdown("---")
            
            if prediction == 1:
                st.error(f"❌ **Risque de défaut ÉLEVÉ** — Probabilité : {proba:.2%}")
                st.write("Ce profil présente des caractéristiques similaires aux clients en défaut. L'octroi du crédit est **déconseillé**.")
            else:
                st.success(f"✅ **Risque de défaut FAIBLE** — Probabilité : {proba:.2%}")
                st.write("Ce profil est jugé solide par le modèle. L'octroi du crédit est **recommandé**.")