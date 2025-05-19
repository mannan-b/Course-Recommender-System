# Course-Recommender-System
A content-based course recommendation system that suggests similar courses based on user input. Built using Python and Streamlit, this application leverages text preprocessing and cosine similarity to provide personalized course recommendations.

🚀 Features
Content-Based Filtering: Recommends courses similar to the user's input by analyzing course descriptions.

Text Preprocessing: Cleans and processes course data to enhance recommendation accuracy.

Interactive Web Interface: User-friendly interface built with Streamlit for seamless interaction.

🗂️ Project Structure
bash
Copy
Edit
Course-Recommender-System/
├── preprocess.py      # Script for data preprocessing
├── recommender.py     # Core recommendation logic
├── streamlit.py       # Streamlit app for user interaction
└── README.md          # Project documentation

⚙️ Setup Instructions
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/mannan-b/Course-Recommender-System.git
cd Course-Recommender-System
Create a Virtual Environment (Optional but recommended):

bash
Copy
Edit
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Application:

bash
Copy
Edit
streamlit run streamlit.py

🧠 How It Works
Data Preprocessing: preprocess.py cleans and prepares the course data for analysis.

Recommendation Engine: recommender.py computes cosine similarity between course descriptions to find similar courses.

User Interface: streamlit.py provides an interactive interface where users can input a course name and receive recommendations.

📌 Usage
Launch the Streamlit app.

Enter the name of a course you're interested in.

View a list of recommended courses similar to your input.
