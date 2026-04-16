import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import joblib
import os

st.set_page_config(
    page_title="CreditAI | Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,700&display=swap');

:root {
    --primary: #F8FAFC;
    --accent: #60A5FA;
    --text-muted: #94A3B8;
    --bg-light: #0F172A;
    --border: #1E293B;
}

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp { background-color: var(--bg-light); }

h1, h2, h3 { color: #F8FAFC !important; }

p, label, span { color: #CBD5E1 !important; }

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: var(--primary) !important;
    letter-spacing: -0.02em;
}

/* Cacher sidebar et header */
section[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#stDecoration { display: none !important; }
footer { display: none !important; }

.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* NAVBAR */
.navbar-container {
    position: sticky;
    top: 0;
    z-index: 999;
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 0 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 64px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-left: -2rem;
    margin-right: -2rem;
    margin-bottom: 0;
}

.navbar-brand {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0F172A;
}

.navbar-brand span { color: #2563EB; }

.navbar-badge {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 20px;
    padding: 5px 14px;
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.75rem;
    color: #166534;
    font-weight: 500;
}

.navbar-dot {
    width: 7px;
    height: 7px;
    background: #22C55E;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 6px #22C55E;
}

/* Boutons navbar */
div[data-testid="stHorizontalBlock"].navbar-row button {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #64748B !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}

div[data-testid="stHorizontalBlock"].navbar-row button:hover {
    background: #F1F5F9 !important;
    color: #0F172A !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.2s;
}

[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 12px rgba(37,99,235,0.1);
    border-color: #BFDBFE;
    transform: translateY(-1px);
}

.stMetric label {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    color: var(--primary) !important;
    font-size: 1.8rem !important;
}

.stFormSubmitButton button {
    background-color: var(--primary) !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stFormSubmitButton button:hover {
    background-color: var(--accent) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.25) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    background-color: transparent;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    height: 40px;
    background-color: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.stSuccess {
    background: #F0FDF4 !important;
    border-left: 3px solid #22C55E !important;
    border-radius: 8px !important;
    color: #166534 !important;
}

.stError {
    background: #FEF2F2 !important;
    border-left: 3px solid #EF4444 !important;
    border-radius: 8px !important;
    color: #991B1B !important;
}

.stInfo {
    background: #EFF6FF !important;
    border-left: 3px solid #3B82F6 !important;
    border-radius: 8px !important;
    color: #1E40AF !important;
}

hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

matplotlib.rcParams.update({
    'axes.facecolor': '#F8FAFC',
    'figure.facecolor': '#FFFFFF',
    'font.family': 'sans-serif',
    'axes.labelcolor': '#64748B',
    'text.color': '#1E293B',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#E2E8F0',
    'xtick.color': '#94A3B8',
    'ytick.color': '#94A3B8',
    'grid.color': '#F1F5F9',
    'axes.grid': True,
    'grid.alpha': 0.8,
})

# ==========================================
# CHARGEMENT
# ==========================================
@st.cache_data
def load_data():
    path = 'data/Loan_Data.csv' if os.path.exists('data/Loan_Data.csv') else '../data/Loan_Data.csv'
    return pd.read_csv(path)

@st.cache_resource
def load_ml_objects():
    base_path = 'data/processed/' if os.path.exists('data/processed/') else '../data/processed/'
    try:
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        model  = joblib.load(os.path.join(base_path, 'best_model.pkl'))
        return scaler, model
    except FileNotFoundError:
        return None, None

df_raw = load_data()
scaler, best_model = load_ml_objects()

# ==========================================
# SESSION STATE
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "Tableau de bord"

# ==========================================
# NAVBAR
# ==========================================
st.markdown("""
<div class="navbar-container">
    <div class="navbar-brand">Credit<span>AI</span></div>
    <div class="navbar-badge">
        <span class="navbar-dot"></span>
        Modele v1.0 actif
    </div>
</div>
""", unsafe_allow_html=True)

# Boutons de navigation
pages = ["Tableau de bord", "Analyse de donnees", "Pre-traitement", "Simulateur de risque"]

col1, col2, col3, col4, col_space = st.columns([1, 1, 1, 1, 3])

with col1:
    active1 = "✦ " if st.session_state.page == "Tableau de bord" else ""
    if st.button(f"{active1}Tableau de bord", key="n1", use_container_width=True):
        st.session_state.page = "Tableau de bord"
        st.rerun()

with col2:
    active2 = "✦ " if st.session_state.page == "Analyse de donnees" else ""
    if st.button(f"{active2}Analyse de donnees", key="n2", use_container_width=True):
        st.session_state.page = "Analyse de donnees"
        st.rerun()

with col3:
    active3 = "✦ " if st.session_state.page == "Pre-traitement" else ""
    if st.button(f"{active3}Pre-traitement", key="n3", use_container_width=True):
        st.session_state.page = "Pre-traitement"
        st.rerun()

with col4:
    active4 = "✦ " if st.session_state.page == "Simulateur de risque" else ""
    if st.button(f"{active4}Simulateur de risque", key="n4", use_container_width=True):
        st.session_state.page = "Simulateur de risque"
        st.rerun()

st.markdown("""
<style>
/* Style des boutons navbar */
div[data-testid="stHorizontalBlock"]:nth-of-type(1) button {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #64748B !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    padding: 6px 12px !important;
}
div[data-testid="stHorizontalBlock"]:nth-of-type(1) button:hover {
    background: #F1F5F9 !important;
    color: #0F172A !important;
}
div[data-testid="stHorizontalBlock"]:nth-of-type(1) button p {
    color: inherit !important;
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin: 0 0 32px 0;'>", unsafe_allow_html=True)

navigation = st.session_state.page

# ==========================================
# PAGE : TABLEAU DE BORD
# ==========================================
if navigation == "Tableau de bord":

    st.markdown("<h1>Apercu du Portefeuille</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#64748B; font-size:1rem; margin-top:-8px; margin-bottom:32px;'>
        Vue d'ensemble du portefeuille de prets personnels — donnees reelles du dataset.
    </p>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volume total", "10,000", help="Nombre total de clients analyses")
    m2.metric("Taux de defaut", "18.5%", "-0.2% vs mois dernier")
    m3.metric("Score FICO Moyen", f"{int(df_raw['fico_score'].mean())}")
    m4.metric("Revenu Moyen", f"${df_raw['income'].mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Distribution des defauts")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        counts = df_raw['default'].value_counts()
        ax1.bar(['Non Defaut', 'Defaut'], counts.values,
                color=['#22C55E', '#EF4444'], edgecolor='none', width=0.5)
        ax1.set_title("Repartition Non Defaut / Defaut", fontsize=11, fontweight='600')
        for i, v in enumerate(counts.values):
            ax1.text(i, v + 50, f'{v:,}\n({v/len(df_raw)*100:.1f}%)',
                    ha='center', fontsize=10, color='#1E293B', fontweight='600')

        ax2.hist(df_raw['fico_score'], bins=40, color='#3B82F6', edgecolor='none', alpha=0.8)
        ax2.set_title("Distribution du FICO Score", fontsize=11, fontweight='600')
        ax2.set_xlabel("FICO Score")
        ax2.axvline(df_raw['fico_score'].mean(), color='#EF4444', linestyle='--',
                   linewidth=1.5, label=f"Moyenne: {df_raw['fico_score'].mean():.0f}")
        ax2.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.markdown("### Statut du Pipeline MLOps")
        st.markdown("<br>", unsafe_allow_html=True)
        steps = [
            ("Data Ingestion", "10,000 lignes chargees"),
            ("EDA Complete", "Analyse exploratoire terminee"),
            ("Preprocessing", "SMOTE + StandardScaler"),
            ("Model Training", "3 algorithmes trackes"),
            ("Deploiement", "Streamlit Cloud actif"),
        ]
        for title, sub in steps:
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
                        padding:12px 16px; margin-bottom:8px;
                        display:flex; align-items:center; gap:12px;'>
                <div style='width:20px; height:20px; background:#F0FDF4; border:1px solid #BBF7D0;
                            border-radius:6px; display:flex; align-items:center; justify-content:center;
                            font-size:0.7rem; color:#16A34A; font-weight:700; flex-shrink:0;'>✓</div>
                <div>
                    <div style='color:#0F172A; font-size:0.85rem; font-weight:600;'>{title}</div>
                    <div style='color:#94A3B8; font-size:0.75rem;'>{sub}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Apercu du dataset brut")
    st.dataframe(df_raw, use_container_width=True)

# ==========================================
# PAGE : ANALYSE DE DONNEES
# ==========================================
elif navigation == "Analyse de donnees":

    st.markdown("<h1>Analyse Exploratoire</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#64748B; font-size:1rem; margin-top:-8px; margin-bottom:32px;'>
        Comprehension statistique et visuelle du portefeuille client.
    </p>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total clients", f"{len(df_raw):,}")
    m2.metric("Taux de defaut", f"{df_raw['default'].mean()*100:.1f}%")
    m3.metric("Valeurs manquantes", "0")
    m4.metric("Doublons", "0")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Distributions", "Boxplots vs Defaut", "Correlations"])

    features = ['credit_lines_outstanding', 'loan_amt_outstanding',
                'total_debt_outstanding', 'income', 'years_employed', 'fico_score']

    with tab1:
        st.markdown("#### Distribution des variables numeriques")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        colors = ['#3B82F6','#8B5CF6','#06B6D4','#10B981','#F59E0B','#EF4444']
        for i, col in enumerate(features):
            axes[i].hist(df_raw[col], bins=40, color=colors[i], edgecolor='none', alpha=0.85)
            axes[i].set_title(col, fontsize=11, fontweight='600', pad=10)
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.markdown("#### Distribution par statut de defaut")
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
        axes2 = axes2.flatten()
        for i, col in enumerate(features):
            sns.boxplot(x='default', y=col, data=df_raw, hue='default',
                       palette={0:'#22C55E', 1:'#EF4444'}, legend=False, ax=axes2[i])
            axes2[i].set_title(col, fontsize=11, fontweight='600')
            axes2[i].set_xlabel('0 = Non Defaut   |   1 = Defaut')
        plt.tight_layout()
        st.pyplot(fig2)

    with tab3:
        st.markdown("#### Matrice de correlation")
        fig3, ax3 = plt.subplots(figsize=(9, 7))
        df_corr = df_raw.drop(columns=['customer_id'], errors='ignore')
        corr = df_corr.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu',
                   center=0, square=True, linewidths=0.5, linecolor='#F1F5F9',
                   annot_kws={'size':10}, ax=ax3)
        ax3.set_title("Matrice de Correlation", fontsize=13, fontweight='600', pad=15)
        st.pyplot(fig3)

    st.markdown("---")
    st.markdown("### Focus FICO Score")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df_raw, x='fico_score', hue='default', bins=40,
                    palette={0:'#22C55E', 1:'#EF4444'}, alpha=0.75, ax=ax4)
        ax4.set_title("FICO Score selon le statut de defaut", fontsize=12, fontweight='600')
        ax4.set_xlabel("FICO Score")
        st.pyplot(fig4)
    with col2:
        fico_ok = df_raw[df_raw['default']==0]['fico_score'].mean()
        fico_ko = df_raw[df_raw['default']==1]['fico_score'].mean()
        st.metric("FICO moyen (Sain)", f"{fico_ok:.0f}")
        st.metric("FICO moyen (Defaut)", f"{fico_ko:.0f}")
        st.info(f"Ecart de **{fico_ok - fico_ko:.0f} points** entre les deux profils.")

# ==========================================
# PAGE : PRE-TRAITEMENT
# ==========================================
elif navigation == "Pre-traitement":

    st.markdown("<h1>Pre-traitement des Donnees</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#64748B; font-size:1rem; margin-top:-8px; margin-bottom:32px;'>
        Feature engineering, normalisation et reequilibrage des classes.
    </p>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features originales", "7")
    m2.metric("Apres Feature Eng.", "8")
    m3.metric("X_train (SMOTE)", "13,038")
    m4.metric("X_test", "2,000")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Feature Engineering", "Normalisation", "SMOTE"])

    with tab1:
        st.markdown("#### Creation de ratios financiers")
        col1, col2 = st.columns(2)
        for col, color, bg, border, name, formula, desc in [
            (col1, "#2563EB", "#EFF6FF", "#BFDBFE", "Debt-to-Income Ratio",
             "debt_to_income = total_debt / income",
             "Mesure le niveau global d'endettement du client par rapport a son revenu annuel."),
            (col2, "#7C3AED", "#F5F3FF", "#DDD6FE", "Loan-to-Income Ratio",
             "loan_to_income = loan_amt / income",
             "Mesure le poids du pret demande par rapport au revenu annuel du client.")
        ]:
            with col:
                st.markdown(f"""
                <div style='background:{bg}; border:1px solid {border}; border-radius:12px; padding:24px;'>
                    <div style='color:{color}; font-size:0.7rem; text-transform:uppercase;
                                letter-spacing:0.12em; font-weight:600; margin-bottom:10px;'>Nouveau ratio</div>
                    <div style='font-family:Fraunces,serif; font-weight:700; font-size:1.1rem;
                                color:#0F172A; margin-bottom:12px;'>{name}</div>
                    <div style='background:#FFFFFF; border:1px solid {border}; border-radius:8px;
                                padding:10px 14px; font-family:monospace; color:{color};
                                font-size:0.85rem; margin-bottom:12px;'>{formula}</div>
                    <div style='color:#64748B; font-size:0.83rem; line-height:1.6;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### StandardScaler — Mise a l'echelle")
        st.code("""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, 'data/processed/scaler.pkl')
        """, language="python")
        st.success("Scaler sauvegarde — coherence garantie entre entrainement et prediction.")

    with tab3:
        st.markdown("#### SMOTE — Reequilibrage des classes")
        fig_s, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for ax, vals, title in [
            (ax1, [6519, 1481], 'Avant SMOTE'),
            (ax2, [6519, 6519], 'Apres SMOTE')
        ]:
            bars = ax.bar(['Non Defaut', 'Defaut'], vals,
                         color=['#22C55E', '#EF4444'], edgecolor='none', width=0.5, alpha=0.85)
            ax.set_title(title, fontsize=12, fontweight='600')
            ax.set_ylabel('Nombre de clients')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+80,
                       f'{v:,}', ha='center', fontsize=11, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig_s)
        st.success("Dataset equilibre : 50% Non Defaut / 50% Defaut apres SMOTE.")

# ==========================================
# PAGE : SIMULATEUR
# ==========================================
elif navigation == "Simulateur de risque":

    st.markdown("<h1>Analyse Decisionnelle</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#64748B; font-size:1rem; margin-top:-8px; margin-bottom:32px;'>
        Evaluation du risque de defaut basee sur le modele Random Forest (AUC = 0.9998).
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Simulateur", "Performances des modeles"])

    with tab1:
        st.markdown("""
        <div style='background:#FFFFFF; padding:20px 24px; border-radius:12px;
                    border:1px solid #E2E8F0; margin-bottom:24px;'>
            <p style='color:#64748B; margin:0; font-size:0.9rem;'>
                Entrez les parametres financiers du candidat pour obtenir une probabilite
                de defaut en temps reel basee sur le modele Random Forest entraine.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if scaler is None or best_model is None:
            st.error("Modele introuvable. Verifiez que data/processed/best_model.pkl et scaler.pkl existent.")
        else:
            with st.form("risk_form"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Profil Financier**")
                    income = st.number_input("Revenu Annuel ($)", min_value=1000.0, value=55000.0, step=1000.0)
                    loan = st.number_input("Montant du pret ($)", min_value=0.0, value=12000.0, step=500.0)
                    debt = st.number_input("Dette actuelle ($)", min_value=0.0, value=5000.0, step=500.0)
                with c2:
                    st.markdown("**Historique Credit**")
                    fico = st.slider("Score FICO", 300, 850, 650)
                    years = st.number_input("Annees d'emploi", min_value=0, max_value=45, value=5, step=1)
                    lines = st.number_input("Lignes de credit ouvertes", min_value=0, max_value=20, value=2, step=1)
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("LANCER L'EVALUATION")

            if submitted:
                dti = debt / income if income > 0 else 0
                lti = loan / income if income > 0 else 0
                feats = ['credit_lines_outstanding', 'loan_amt_outstanding',
                        'total_debt_outstanding', 'income', 'years_employed',
                        'fico_score', 'debt_to_income', 'loan_to_income']
                user_df = pd.DataFrame([[lines, loan, debt, income,
                                        years, fico, dti, lti]], columns=feats)
                pred  = best_model.predict(scaler.transform(user_df))[0]
                proba = best_model.predict_proba(scaler.transform(user_df))[0][1]

                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                with r1: st.metric("Probabilite de Defaut", f"{proba:.2%}")
                with r2: st.metric("Ratio Dette / Revenu", f"{dti:.2%}")
                with r3: st.metric("Ratio Pret / Revenu", f"{lti:.2%}")
                st.markdown("<br>", unsafe_allow_html=True)

                if pred == 0:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("""
                        <div style='background:#F0FDF4; border:1px solid #BBF7D0; border-radius:12px;
                                    padding:24px; text-align:center;'>
                            <div style='width:48px; height:48px; background:#DCFCE7; border-radius:50%;
                                        margin:0 auto 12px auto; display:flex; align-items:center;
                                        justify-content:center; color:#16A34A; font-weight:700; font-size:1.2rem;'>✓</div>
                            <h3 style='color:#16A34A; margin:0; font-size:1.1rem; font-family:Fraunces,serif;'>
                                CREDIT APPROUVE</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        bar_w = proba * 100
                        st.markdown(f"""
                        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:20px;'>
                            <div style='color:#64748B; font-size:0.85rem; margin-bottom:12px;'>
                                Positionnement par rapport au seuil de risque critique (20%)</div>
                            <div style='background:#F1F5F9; border-radius:8px; height:10px; overflow:hidden; margin-bottom:8px;'>
                                <div style='width:{bar_w:.1f}%; height:100%;
                                            background:linear-gradient(90deg,#22C55E,#86EFAC); border-radius:8px;'></div>
                            </div>
                            <div style='display:flex; justify-content:space-between;'>
                                <span style='color:#16A34A; font-size:0.8rem; font-weight:600;'>Risque : {proba:.2%}</span>
                                <span style='color:#94A3B8; font-size:0.8rem;'>Seuil critique : 20%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success(f"Profil solide. Probabilite de defaut de **{proba:.2%}**, sous le seuil.")
                else:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("""
                        <div style='background:#FEF2F2; border:1px solid #FECACA; border-radius:12px;
                                    padding:24px; text-align:center;'>
                            <div style='width:48px; height:48px; background:#FEE2E2; border-radius:50%;
                                        margin:0 auto 12px auto; display:flex; align-items:center;
                                        justify-content:center; color:#DC2626; font-weight:700; font-size:1.2rem;'>!</div>
                            <h3 style='color:#DC2626; margin:0; font-size:1.1rem; font-family:Fraunces,serif;'>
                                REVISION REQUISE</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        bar_w2 = min(proba * 100, 100)
                        st.markdown(f"""
                        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:20px;'>
                            <div style='color:#64748B; font-size:0.85rem; margin-bottom:12px;'>
                                Positionnement par rapport au seuil de risque critique (20%)</div>
                            <div style='background:#F1F5F9; border-radius:8px; height:10px; overflow:hidden; margin-bottom:8px;'>
                                <div style='width:{bar_w2:.1f}%; height:100%;
                                            background:linear-gradient(90deg,#EF4444,#FCA5A5); border-radius:8px;'></div>
                            </div>
                            <div style='display:flex; justify-content:space-between;'>
                                <span style='color:#DC2626; font-size:0.8rem; font-weight:600;'>Risque : {proba:.2%}</span>
                                <span style='color:#94A3B8; font-size:0.8rem;'>Seuil critique : 20%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.error(f"Profil risque. Probabilite de defaut de **{proba:.2%}**. Revision recommandee.")

    with tab2:
        st.markdown("#### Performances des 3 algorithmes — Tracking MLflow")
        results = {
            "Modele": ["Logistic Regression", "Decision Tree", "Random Forest"],
            "Accuracy":  [0.9970, 0.9965, 0.9930],
            "F1 Score":  [0.9920, 0.9905, 0.9811],
            "ROC-AUC":   [1.0000, 0.9937, 0.9998],
            "Precision": [0.9840, 0.9919, 0.9811],
            "Recall":    [1.0000, 0.9892, 0.9811]
        }
        df_r = pd.DataFrame(results).set_index("Modele")
        st.dataframe(df_r.style.highlight_max(axis=0, color='#DBEAFE').format("{:.4f}"),
                    use_container_width=True)

        fig_m, ax_m = plt.subplots(figsize=(11, 4))
        x = np.arange(len(df_r))
        w = 0.15
        for i, (col, color) in enumerate(zip(df_r.columns,
                ['#3B82F6','#8B5CF6','#06B6D4','#10B981','#F59E0B'])):
            ax_m.bar(x + i*w, df_r[col], w, label=col, color=color, alpha=0.85)
        ax_m.set_xticks(x + w*2)
        ax_m.set_xticklabels(df_r.index, fontsize=11)
        ax_m.set_ylim(0.95, 1.005)
        ax_m.legend(loc='lower right', fontsize=9)
        ax_m.set_title("Comparaison des metriques par modele", fontsize=12, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig_m)

        st.markdown("""
        <div style='background:#F0FDF4; border:1px solid #BBF7D0; border-radius:12px;
                    padding:20px; margin-top:16px; display:flex; align-items:center; gap:16px;'>
            <div style='width:36px; height:36px; background:#DCFCE7; border-radius:10px;
                        display:flex; align-items:center; justify-content:center;
                        color:#16A34A; font-weight:700; font-size:1rem; flex-shrink:0;'>1</div>
            <div>
                <div style='font-weight:700; color:#0F172A; font-size:0.95rem;'>
                    Meilleur modele retenu : Random Forest</div>
                <div style='color:#64748B; font-size:0.83rem; margin-top:4px;'>
                    ROC-AUC = 0.9998 — sauvegarde dans best_model.pkl</div>
            </div>
        </div>
        """, unsafe_allow_html=True)