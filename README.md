# NetGainNBA

This project is a web application that uses ML to predict NBA match outcomes. It combines **data collection, cleaning, machine learning modeling, and simulations** to provide actionable insights for front office management.  

---

## ML Workflow

1. **Data Collection & Cleaning**  
   - Fetched team, player, and matchup data from the NBA API.  
   - Joined and cleaned datasets to create structured inputs for modeling.  

2. **Player & Team Clustering**  
   - Clustered players into playstyles using **K-Means clustering**.  
   - Transformed player clusters into **team-level playstyle proportions**.  

3. **Regression Modeling**  
   - Built **Random Forest** and **XGBoost** models to forecast the number of playoff wins for a team.  

4. **Classification Modeling**  
   - Shifted to a **classification approach** using **TabNet neural networks**.  
   - Used **temporal sliding cross-validation** to predict actual playoff matchups.  
   - Achieved **88% playoff prediction accuracy (2020–2024)**.  

5. **Simulation & Monte Carlo Analysis**  
   - Introduced **controlled randomness** via Monte Carlo simulations.  
   - Predicted playoff winners and displayed **round-by-round probabilities** for each team.  

**Application Insight:**  
- Front office management can analyze which playstyles are most impactful for playoff performance.  
- Users can simulate **any playoff bracket** and visualize the effect of different team archetypes.  

---

## Getting Started

To run this project (FOR NOW):

### NOTE: NEED TO MAKE A .env FILE AND PUT ALL THE AUTHENTICATION KEYS FOR THIS TO WORK

1. Navigate to the frontend folder:

   ```
   cd frontend
   ```

2. Install dependencies:

   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

## Backend Setup

To set up the backend environment (you can skip directly to the pip install stuff if you don't want to set up a virtual enviorment):

### macOS

1. **Navigate to the backend directory:**

   ```bash
   cd backend
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

4. **Install the required dependencies:**

   ```bash
   pip install Flask Flask-CORS pandas
   ```

5. **Run the backend server:**

   ```bash
   python main.py
   ```

### Windows

1. **Navigate to the backend directory:**

   ```cmd
   cd backend
   ```

2. **Create a virtual environment:**

   ```cmd
   python -m venv .venv
   ```

3. **Activate the virtual environment:**

   ```cmd
   .venv\Scripts\activate
   ```

4. **Install the required dependencies:**

   ```cmd
   pip install Flask Flask-CORS pandas
   ```

5. **Run the backend server:**

   ```cmd
   python main.py
   ```
