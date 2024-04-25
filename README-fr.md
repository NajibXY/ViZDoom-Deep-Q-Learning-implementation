# Implémentation d'un algorithme d'apprentissage par renforcement profond dans l'environnement VizDoom

## Auteur: [Najib El khadir](https://github.com/NajibXY)

## 1. Motivations

<figure text-align="right">
  <img align="right" src="https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/assets/basic_conf_fullscreen.gif" width="400">
</figure>

Durant ma deuxième année en master informatique spécialisé en Intelligence Artificielle, j'ai eu l'occasion d'entendre parler d'une bibliothèque 
Python fournissant des environnements d'entrainement pour agents apprenants : VizDoom. Dans le cadre donc de mes expérimentations d'Avril 2024, j'ai
entrepris d'implémenter un algorithme d'apprentissage par renforcement profond et de le tester sur des environnements de  VizDoom.

</br> </br>

## 2. Technologies Utilisées
![](https://skillicons.dev/icons?i=python,pytorch,anaconda)
- Python 3.12, PyTorch, VizDoom, Conda (pour mon environnement personnel)

## 3. Références
- [Scénarios et environnements VizDoom](https://vizdoom.farama.org/environments/default/)
- [Exemple basique de la fondation FARAMA sur l'utilisation de VizDoom](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/learning_pytorch.py)

## 4. Génération de données

- Les points de données du Mandelbulb sont assez faciles à générer.
- Mais pour pouvoir afficher ces données, une méthode de ray marching a été implémentée, qui implique le calcul des lumières et des distances selon le mouvement d'une caméra virtuelle. Cela nécessite de faire de la parallélisation avec OpenMP et beaucoup d'algèbre vectorielle tridimensionnelle pour obtenir un temps d'exécution acceptable.
- Il faut environ 30 minutes pour générer les données avec un exposant de base égal à 7.

### Compilation du fichier generate_data.cpp

- Nécessite C++ et gcc/g++ installés.
- Exécutez simplement :
  > g++ -Wall -Wextra -fopenmp generate_data.cpp -o generate_data.exe
- L'exécution de l'.exe générera les données dans le dossier /data/.

## 5. Affichage des données de rendu

### Configuration de votre environnement

- Vous devrez configurer un environnement Python (dans mon cas conda) pour afficher les données générées avec Matplotlib.
- Pour installer les dépendances avec conda, vous pouvez simplement exécuter :
  > conda create --name `<votre_nom_env>` --file requirements.txt
- Ou si vous utilisez pip :
  > pip install -r requirements.txt
- Ensuite, vous pourrez exécuter le script.

### Exécution du script Python

> python .\generate_video.py
- Cela générera tous les PNG rendus à partir des données précédemment générées. Ces images sont enregistrées dans le dossier /images/.
- De plus, le script compile ces images en une vidéo à l'aide de FFMPEG et la sauvegarde dans le même dossier que le script.
</br>

### Exemples d'images et de vidéo générées 

+ Exemple d'une vidéo générée :
   </br>
   </br>
  <img src="https://github.com/NajibXY/Mandelbulb-with-Ray-marching/blob/master/assets/mandelbulb.gif" width="400">
  </br>
+ Exemples d'images générées et colorées aléatoirement :
  </br>
  </br>
  <img src="https://github.com/NajibXY/Mandelbulb-with-Ray-marching/blob/master/assets/example1.png" width="350">
  </br>
  <img src="https://github.com/NajibXY/Mandelbulb-with-Ray-marching/blob/master/assets/example2.png" width="350"> 

</br></br>

## 5. Améliorations Possibles

- Génération de données plus rapide.
- Utilisation d'autres formules pour la génération de données.
- Utilisation d'autres méthodes de rendu.
- Utilisation d'une bibliothèque C++ efficace pour la visualisation des données conjointement à leurs calculs.
- Fournir la possibilité de tuner les paramètres de génération.
- [...]
