"""
📝 **Instructions** :
- Installez toutes les bibliothèques nécessaires en fonction des imports présents dans le code, utilisez la commande suivante :conda create -n projet python pandas numpy ..........
- Complétez les sections en écrivant votre code où c’est indiqué.
- Ajoutez des commentaires clairs pour expliquer vos choix.
- Utilisez des emoji avec windows + ;
- Interprétez les résultats de vos visualisations (quelques phrases).
"""
# conda create -n projet python pandas numpy matplotlib seaborn streamlit plotly 
# conda activate projet

### 1. Importation des librairies et chargement des données
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import kagglehub
from pathlib import Path
from PIL import Image

#st.set_page_config(layout="wide")
path = kagglehub.dataset_download("arnabchaki/data-science-salaries-2023")

st.sidebar.write("Créé par : Sergina Bakala et Kiran Derennes",
                "BUT 3 Sciences des Données - Groupe D",
                "Année Universitaire 2024-2025")

print("Path to dataset files:", path)
# Chargement des données
df = pd.read_csv(Path(path) / "ds_salaries.csv")

### 2. Exploration visuelle des données
#votre code 
st.title("📊 Visualisation des Salaires en Data Science")
st.markdown("Explorez les tendances des salaires à travers différentes visualisations interactives.")


if st.checkbox("Afficher un aperçu des données"):
    st.write(df.head(5))


#Statistique générales avec describe pandas 
#votre code 
st.subheader("📌 Statistiques générales ")
if st.checkbox("Afficher les statistiques générales des données"):
    st.write(df.describe())

### 3. Distribution des salaires en France par rôle et niveau d'expérience, uilisant px.box et st.plotly_chart
#votre code 
st.subheader("📈 Distribution des salaires en France")
df_france = df.loc[df["company_location"] == "FR"]
#st.write(df_france.head(5))
fig = px.box(df_france, x="job_title", y="salary_in_usd", color='experience_level')  # Créer un boxplot interactif
st.plotly_chart(fig)
st.write("Interprétation du graphique:")
st.write("""
         Dans ce graphique on peut voir que les employés avec un niveau d'expérience sénior ont en moyenne un salaire plus élevé que les employés au niveau entry.
         De plus, les machine learning infrastructure engineers et les machine learning engineers ont une fourchette de salaire similaire entre 75 mille dollards et plus de 100 mille dollards.
         """)

### 4. Analyse des tendances de salaires :
#### Salaire moyen par catégorie : en choisisant une des : ['experience_level', 'employment_type', 'job_title', 'company_location'], utilisant px.bar et st.selectbox 
st.subheader("📈 Analyse des tendances de salaires")
option = st.selectbox("Choisissez une catégorie:",('experience_level', 'employment_type', 'job_title', 'company_location'))
df_salaire_moyen_option = df.groupby(option)["salary_in_usd"].mean().reset_index().sort_values(["salary_in_usd"])
fig=px.bar(df_salaire_moyen_option,x= option, y = "salary_in_usd")

st.plotly_chart(fig)

st.write("Interprétation du graphique:")
st.write("""
         Le graphique ci-dessus confirme l'interprétation du graphique précédent, les employés de niveau sénior sont en moyenne mieux payé que les employés à l'entry level.
         Mais les experts sont mieux payés que tous les autres employés en moyenne.
         lors de la séléction du filtre sur les types d'employés dans les entreprises, on constate que les employés "FT" sont ceux qui ont un salaire le plus élevé par rapport au "PT"
         """)
### 5. Corrélation entre variables
# Sélectionner uniquement les colonnes numériques pour la corrélation
numeric_df = df.select_dtypes(include=[np.number])  # Sélectionner uniquement les colonnes numériques
# Calcul de la matrice de corrélation
#votre code
matrice = numeric_df.corr()
# Affichage du heatmap avec sns.heatmap
#votre code 
st.subheader("🔗 Corrélations entre variables numériques")
plot = sns.heatmap(matrice, annot=True, cmap='coolwarm')
st.pyplot(plot.get_figure())
st.write("Interprétation du graphique:")
st.write("""
         Ce graphique présente la corrélation entre les différentes variables du jeu de données, nous pouvons voir qu'il n'y a pas de corrélation de manière globale entre les variables ( car très éloignés de -1 et 1).
On peut donc dire que les variables évoluent de manière indépendanteset qu
         """)
### 6. Analyse interactive des variations de salaire
# Une évolution des salaires pour les 10 postes les plus courants
# count of job titles pour selectionner les postes
# calcule du salaire moyen par an
#utilisez px.line
#votre code 

top10_poste = df.groupby('job_title')['job_title'].count().sort_values(ascending=False).head(10).index
df_top10 = df[df['job_title'].isin(top10_poste)]
df_top10=df_top10.groupby(["work_year","job_title"])["salary_in_usd"].mean().reset_index()
graph_sal_moy_an = px.line(df_top10, x='work_year', y='salary_in_usd', color='job_title')

st.plotly_chart(graph_sal_moy_an.update_layout(xaxis_title="Année", yaxis_title="Salaires moyen en USD"))
st.write("Interprétation du graphique:")
st.write("""
         Ce graphique présente l'évolution du salaire sur plusieurs années en fonction des jobs titles les plus courant. Il est difficile de déceler une tendance globale pour tous les job title.
         Par contre nous pouvons observé qu 2021 beaucoup de salaires ont été baissé surement explicable par le virus.
         Mais plus récement le job_title avec le salaire le plus élévé est Data science manager, à l'inverse le data analyste à un salaire moins élevé que ses paires en moyenne.
         Ce graphique présente aussi l'apparition de métier en 2021 ou 2022 comme Data architect ou Research engineer.
         """)
### 7. Salaire médian par expérience et taille d'entreprise
# utilisez median(), px.bar
#votre code 
sal_med = df.groupby(["experience_level","company_size"])["salary_in_usd"].median().reset_index()
fig = px.bar(sal_med, x="company_size", y="salary_in_usd", color="experience_level")  # Créer un boxplot interactif
st.plotly_chart(fig)
st.write("Interprétation du graphique:")
st.write("""
         Le graphique ci-dessus présente la répartition des salaires en fonction de la taille de l'entreprise différentié par le niveau d'expérience.
         Ainsi les employés d'une entreprise moyenne ont un salaire plus élevé que les employés d'une entreprise plus petite ou grande.
         Les experts ont en moyenne une grande fourchette de salaire comparé aux autres niveaux d'employés.
         """)

### 8. Ajout de filtres dynamiques
#Filtrer les données par salaire utilisant st.slider pour selectionner les plages 
#votre code 

salaire = st.slider("Plage de salaire",value = (df["salary_in_usd"].min(), df["salary_in_usd"].max()))
st.write(salaire[0])
df_filtrer_sal = df.loc[df["salary_in_usd"].between(salaire[0],salaire[1])]
st.write(df_filtrer_sal.head(5))
st.write(len(df_filtrer_sal))

### 9.  Impact du télétravail sur le salaire selon le pays
salaire_pays = df.groupby(["company_location","remote_ratio"])["salary_in_usd"].mean().reset_index()
fig = px.bar(salaire_pays, x = "company_location", y = "salary_in_usd", color= "remote_ratio")
st.plotly_chart(fig)

st.write("Interprétation du graphique:")
st.write("""
         Ce graphique présente la différence salariale selon l'action de faire du télétravail en fonction des pays.
         Les employés d'une entreprise avec un taux de télétravail plus élevé ont en générale et en moyenne un salaire plus bas que ceux avec un taux de télétravail bas, 
         par exemple pour le pays 'ID', les employés avec un taux de télétravail de 100% sont payés en moyenne 119 milles dollards tandis que ceux avec un taux de télétravail bas sont payés 423 milles dollards. 
         On observe par contre un phénomène contraire dans certains pays d'Europe comme 'FR' où la réalisation du télétravail donne droit à des indmnités ce qui explique pourquoi ce sont cette fois-ci les télétravailleurs qui sont en moyenne mieux payés que les non-télétravailleurs.
         """)

### 10. Filtrage avancé des données avec deux st.multiselect, un qui indique "Sélectionnez le niveau d'expérience" et l'autre "Sélectionnez la taille d'entreprise"
#votre code 
niveau_xp = st.multiselect("Sélectionnez le niveau d'expérience", df["experience_level"].unique())
taille_ent = st.multiselect("Sélectionnez la taille d'entreprise", df["company_size"].unique())
df_filter_xp_ent = df.loc[df["experience_level"].isin(niveau_xp) | df["company_size"].isin(taille_ent)]
st.write(df_filter_xp_ent.head(5))
st.write(len(df_filter_xp_ent))
