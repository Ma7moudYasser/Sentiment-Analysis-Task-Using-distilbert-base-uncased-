# Sentiment-Analysis-Task-Using-distilbert-base-uncased
# Project documentation
#Introduction
This project focuses on sentiment analysis using the IMDb dataset and fine-tuning a DistilBERT-based model for binary classification (positive or negative sentiment). The primary objective is to develop a robust sentiment analysis model and showcase the fine-tuning process using Hugging Face's Transformers library.

#Data Description
The IMDb dataset consists of movie reviews labeled as either positive or negative sentiments. The dataset is divided into training and testing sets, allowing for model evaluation on unseen data. The features include textual content from the movie reviews.


#Baseline Experiments
The goal of the baseline experiments is to establish a foundation for sentiment analysis using a pre-trained DistilBERT model. The model is loaded and fine-tuned on the IMDb dataset, and the training process is evaluated using accuracy as the primary metric. The baseline experiments aim to provide insights into the initial performance of the model.


#Fine-Tuning the Model
The fine-tuning process involves training the DistilBERT model on the IMDb dataset with specific configurations, such as learning rate to be 2e ^ -5, batch size = 32, and number of epochs = 2. The goal is to optimize the model's performance on sentiment analysis tasks. The fine-tuning steps are outlined in the code, showcasing the model's adaptation to the sentiment analysis domain.


## Please note that the parameters can be optimized and the batch size can be decreased but I encountered the problem of lacking resources and the model took about 3 hours to get those results 

## Tools used 

###Transformers library
###Datasets library
###Evaluate library
###Accelerate library
###Hugging Face Model Hub
###Notebook login from Hugging Face Hub

# Please note that the model solved the extreme cases sent inside the task 
### You can test the [![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model%20Link-blue)](https://huggingface.co/Medo3110/my_awesome_model/commit/9daa469c1a3f1f052dc89f4c91ff505fae940dc4)



# Test cases results 

![WhatsApp Image 2024-01-22 at 4 27 42 AM](https://github.com/Ma7moudYasser/Sentiment-Analysis-Task-Using-distilbert-base-uncased-/assets/57537704/8036f78e-501b-4609-806b-4d635d73c6cc)


# Model performance 




![WhatsApp Image 2024-01-22 at 4 26 43 AM](https://github.com/Ma7moudYasser/Sentiment-Analysis-Task-Using-distilbert-base-uncased-/assets/57537704/bf669afc-b4fd-4258-a539-e19d51056e07)
