# NBA Net Gain Web Application

This project is a web application that uses ML to predict NBA match outcomes

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
