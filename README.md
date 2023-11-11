
References Kaggle Fake and Real News Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
How to Run the Program
1. Clone the Repository:
   - Open a terminal or command prompt.
   - Run the following command to clone the repository:
     git clone https://github.com/Jagan6923/Fake-News-Detection-Using-NLP.git

2. Navigate to the Project Directory:
   - Change into the project directory:
     cd your-repository

3. Install Dependencies:
   - Make sure you have Python installed on your system.
   - Install the required dependencies using the following command:
     pip install -r requirements.txt

4.Dependencies
	1.Pandas
	2.Re
	3.Nltk
	4.scikit-learn
	5.matplotlib
	6.seaborn

5. Run the Code:
   - Execute the main program file, which appears to be named fake_news_detection.py:
     python fake_news_detection.py
    
6. Download NLTK Stopwords (First Run):
   - During the first run, the program will download the NLTK stopwords dataset. Ensure you have an active internet connection for this step.

7. View the Results:
   - The program will print the following metrics:
     - Accuracy
     - Precision
     - Recall
     - F1 score
     - ROC-AUC score
   - A confusion matrix will be displayed in a heatmap.

8. Adjusting Parameters (Optional):
   - If you want to experiment with different parameters or algorithms, you can modify the code within fake_news_detection.py.
   - Consider exploring other machine learning algorithms or adjusting hyperparameters for potential improvements.

Note:
- Ensure you have an active internet connection during the first run to download the NLTK stopwords dataset.
- If you encounter any issues, check the console output for error messages, and refer to the documentation or seek help online.
