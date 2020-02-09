# Text-Generation-RNN
RNN with attention layer used for text-generation.

# Description
This project enables training of a text-generating neural network of a selected size on a given text dataset.

# Architecture 

![Screenshot](static/images/default_model.png)

# Usage
**not required
## Docker
  
1) docker build -t [CONTAINER_NAME] [PATH_TO_DOCKERFILE](if it is in Your current directory use ".")  
2) docker run [CONTAINER_NAME]  
3) By deafault Your docker should run on 172.17.0.2:5000. But If it is not working You have to check  
it by using some additinal commands  
4) docker ps -> check Your's container name (they are random generated, last column) **
5) docker inspect (name from previous command)  **
6) find "IPadress". It should be something like 172.17.0.* **
7) Now You just need to paste this IP with proper port (5000 by default) in Your browser **


## Without Docker(python3, virtualenv recommended) 
1) git clone https://github.com/Piotrekszmel/Text-Generation-RNN.git
2) cd Text-Generation-RNN
1) pip3 install -r requirements
2) python3 app.py
3) by default app should start on http:0.0.0.0:5004, so just go to Your browser and paste this URL

 

