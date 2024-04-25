# Implémentation d'un algorithme d'apprentissage par renforcement profond dans l'environnement VizDoom

## Auteur: [Najib El khadir](https://github.com/NajibXY)

## 1. Motivations

<figure text-align="right">
  <img align="right" src="https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/assets/basic_conf_fullscreen.gif" width="400">
</figure>

Durant ma deuxième année en master informatique spécialisé en Intelligence Artificielle, j'ai eu l'occasion d'entendre parler d'une bibliothèque 
Python fournissant des environnements d'entraînement pour agents apprenants : VizDoom. Dans le cadre donc de mes expérimentations d'avril 2024, j'ai
entrepris d'implémenter un algorithme d'apprentissage par renforcement profond et de le tester sur des environnements de  VizDoom.

</br> </br>

## 2. Technologies Utilisées
![](https://skillicons.dev/icons?i=python,pytorch,anaconda)
- Python 3.12, PyTorch, VizDoom, Conda (pour mon environnement personnel)

## 3. Références
- [Scénarios et environnements VizDoom](https://vizdoom.farama.org/environments/default/)
- [Exemple basique de la fondation FARAMA sur l'utilisation de VizDoom](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/learning_pytorch.py)
- [Qu'est ce que l'apprentissage par renforcement profond](https://www.v7labs.com/blog/deep-reinforcement-learning-guide)

## 4. Fichier [DQL.py](https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/dql.py)

- L'implémentation est entièrement faite dans le fichier [DQL.py](https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/dql.py) pour l'instant.
</br></br>
- Ce script se décompose comme suit :
  
### Paramètres de l'environnement
- Définition des configurations VizDoom à charger.
- L'environnement étant perçu par l'agent comme un ensemble de pixel, il faut définir une résolution qui sera utilisé dans VizDoom et une fenêtre de répétition.

### Replay Memory
- Une classe Replay Memory est implémentée.
- Cette classe est utilisée pour stocker les transitions récentes (état source, action, état résultat, récompense).
- Cette classe permet aussi de tracer les actions terminales (mettant fin à l'épisode courant).

### Classe QNN
- Cette classe est le coeur de l'algorithme d'apprentissage profond.
- Elle permet de créer un réseau de neurones convolutif avec 1 couche d'entrée, 1 couche de sortie et 2 couches principales.
- Elle permet à l'agent de sélectionner la meilleure action à une étape donnée, de jouer une étape et d'adapter son comportement en fonction de la Replay Memory.

### Fonctions utiles
- Il y a également des fonctions permettant le traitement de l'image courante de la fenêtre VizDoom, d'initialiser la simulation et de connaitre son état.
- Une fonction est aussi dédiée au déroulement d'épisodes de démo une fois l'apprentissage terminé. Cela nécessite de changer les paramètres adéquats dans la boucle principale du script.
- La partie FLAGS de la fonction principale permet aussi de tuner les paramètres d'entraînement (nombre d'épisodes, nombre d'itérations par épisode, taille du batch, taux d'apprentissage, etc.). N'hésitez pas à expérimenter avec cette partie.

## 5. Entraînement de l'agent

### Configuration de votre environnement

- Vous devrez configurer un environnement Python (dans mon cas conda), de préférence 3.12 pour éviter les problèmes de compatibilité de bibliothèques.
- Pour installer les dépendances avec conda, vous pouvez simplement exécuter :
  > conda create --name `<votre_nom_env>` --file requirements.txt
- Ou si vous utilisez pip :
  > pip install -r requirements.txt
- Ensuite, vous pourrez exécuter le script.

### Exécution du script Python

> python .\dql.py
- Cela lancera l'entraînement de l'agent sur l'environnement `basic.cfg`
- Les résultats de l'entraînement sont stockés dans un fichier `saved_model_doom.pth`
- Une fois l'entraînement terminé vous pouvez dérouler des tests sur votre agent entrainé en mettant les FLAGS "skip_training" et "load_model" à True dans la fonction principale du script. 
</br></br>
- Après plusieurs tests avec différents paramétrages, voici quelques résultats "satisfaisants" qui ont été obtenus.
  
### Exemples 

+ Exemple d'un agent après 20 épisodes de 2000 itérations sur un environnement `basic.cfg` 

  <img src="https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/assets/basic_conf_fullscreen.gif" width="400">
  </br>
    - L'agent a effectivement bien appris à se déplacer de droite à gauche et réagir à son environnement au bon moment en tirant sur la cible. 
  </br>
  
+ Exemple d'un agent après 20 épisodes de 2000 itérations sur un environnement `defend_the_center.cfg` 

  <img src="https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/assets/defend_the_center.gif" width="350">
  </br>
  - Ici l'agent a eu plus de mal à trouver une stratégie humainement fiable.
  </br>  
  - Néanmoins on peut observer que le comportement inféré est plutôt cohérent : défendre le milieu en tournant et tirant (stratégie du spam !?)

</br></br>

## 5. Améliorations Possibles

- Tuner l'algorithme pour mieux l'adapter aux environnements différents.
- Continuer les expérimentations sur les autres environnements.
- Expérimenter avec d'autres modèles d'apprentissage profond par renforcement.
- Développer un GUI ou un CLI pour permettre d'entrer les paramètres et la méthode d'apprentissage, l'environnement, etc.
- [...]
