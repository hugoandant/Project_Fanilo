# Utiliser l'image existante de Jupyter Notebook comme base
FROM quay.io/jupyter/pyspark-notebook

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /tmp/

# Installer les dépendances Python spécifiées dans le fichier requirements.txt
RUN pip install -r /tmp/requirements.txt