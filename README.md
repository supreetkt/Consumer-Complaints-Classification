# Consumer-Complaints-Classification
This is an NLP-based problem solving approach for the dataset available as a consumer-complaint database for the Banking sector. It contains 1,179,715 rows and 18 columns. The dataset contains 300k+ rows of complaints texts.

## Setting up the repository
Github has a file limit size of 100MB and I pre-trained models and saved them so that I wouldn’t have to run them every time. Pre-processing of each row to clean the text specifically took a lot of time, so it has a corresponding csv with it.

So, before running anything, set up the repo and complete the missing files. The architecture should look like this:

Consumer-Complaints-Classification Project
1. app FOLDER
- contains services, static and templates. This is for flask. Also contains main.py, used to run the application.

2. data FOLDER
- contains the main dataset and a subfolder modified. Modified folder contains all the intermediary csvs used to reduce processing each time the code is run.
- vocab_trained_word2Vec.csv - contains the output of word2vec model
- output_consumer_complaints.csv - contains the preprocessed and cleaned set of words using the clean_up function

3. trained_models FOLDER
- Contains GoogleNews-vectors-negative300.bin, which can be downloaded online through the link I provided
- 300features_10minwords_10context  (attached in google drive) - word2Vec model created by using the training set
- rf_word2vec_model_6530 (attached in google drive) - The random Forest model, which uses word2Vec and has an accuracy of 65.30%

4. src  FOLDER - contains one notebook and one py file. Both have the same code.

- For any help in running the code, shoot me an email. I’ll be happy to help.


## Instructions to run the Flask based GUI:
You can watch the video flask_prediction.mp4 in the main folder to help you see how the Flask component work. For some reason, this was working perfectly in my office's Windows PCs, but giving errors in my personal Mac. I read online that it is due to outdated numpy libraries. I updated it on my Mac, but it didn't help. At the same time, on my Windows PC, it didn't give any issues, which is where I recorded this video.

1. Download the pre-trained Google model from the link provided and place it in trained_models FOLDER. 

2. Download the main dataset’s csv file from the link: https://catalog.data.gov/dataset/consumer-complaint-database. Place it in the data FOLDER.

Note that intermediary files have been uploaded to help run the file. For example, if a model takes a long time to train, its output is saved under data/modified and will be used during the code run instead of freshly making new files.
Models have been saved for efficiency, such as the word2Vec model and the Random Forest Model which turns up the winner in the code.

3. Go to the app folder where the main.py file exists

4. Open a terminal. cd to this path/folder.

3. Type: 'python main.py' on the terminal.

A web browser would open up and show you text-boxes to enter information. The complaints are classified in one of 18 classes, each of which represent a category/product-type, such as Mortgage, Student Loans, Credit Reporting, etc. The text would be classified into one out of these 18 classes and the result would be shown on the next page alongwith a probability score associated with that class label.

## Examples of input types to the Flask GUI
Example texts for testing the model (can be seen in the dataset; bigger strings take slightly more processing time):


1. I 'm XXXX now and I 'm trying to get my free credit report and I keep getting my information does n't match my record? 
- Credit Reporting type category

2. I '' m a victim of fraud and I have a file with the Federal Trade Commission. While preparing to check my credit for incorrect information, I have noticed there are six inquires that I have no knowledge of nor do I have or ever had any accounts opened with the companies, please will you help me in solving these issues? 
 1.XXXX/XXXX ... .National Credit Cards/Airlines 4.XXXX XXXX .... 
 inquiry date : XXXX inquiry date : XXXX 2.XXXX/XXXX .... 5. XXXX XXXX .... 
 inquiry date : XX/XX/XXXX inquiry date : XXXX 3.XXXX .... 6.  XXXX XXXX ... 
 inquiry date : XX/XX/XXXX inquiry date : XX/XX/XXXX
- Credit Card

3. Slightly lengthy - "My mortgage is with BB & T Bank, recently I have been investigating ways to pay down my mortgage faster and I came across Biweekly Mortgage Calculator on BB & T 's website. It's a nice, easy to use calculator that you plug in your interest rate, mortgage amount, mortgage term, and payment type and it calculates your accelerated bi-weekly payment for you and shows you how much quicker you can pay down your loan. Ours figured out to pay off a 30 year mortgage in 26.4 years ... quite a savings! 
I called BB & T 's customer service number to inquire how I get set up on this payment plan. I was told they do not offer that type of payment plan, but I could send in my payments bi-weekly but it would not be applied until the full amount was received. ( the money would sit in a "" holding account '' until the full payment amount was collected ). I ended up calling back a few days later thinking the rep I was talking to didn't understand what I wanted to do or was not knowledgeable of this program. I got the SAME ANSWER! 
I then asked for the corporate BB & T office number where I could speak to someone that was knowledgeable of this product. After 3 days I received a phone call back from a corporate manager stating they do not offer this product, and they were "" checking into why this is on their website ''. She stated they do have a few customers that make bi-weekly payments, but they no longer offer this service. 
I don't understand how they can have this active link on their website under their Financial Planning Center tab to mislead customers when all they say is "" I'm sorry, I know you're upset about this '' Sounds like false advertising to me! 
- Mortgage category

4. I applied for credit with Synchrony Bank XXXX FL and was turn down because their system said my credit score is too low [ XXXX ] this is not correct my score is XXXX XXXX and XXXX XXXX I feel this is Discrimination
- Credit card category

5. I received a copy of my credit report and there was a collection reported from XXXX XXXX XXXX but the bill was paid directly to the provider. There was a mix up with the insurance and it wasn't until a little later I was informed that my insurance coverage did not pay for it so I paid immediately. This never should have been sent to collections it was paid before this even started to report.
- Debt collection










