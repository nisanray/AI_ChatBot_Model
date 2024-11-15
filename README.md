# Weather and Chatbot Application

## Features

- **Weather Data Retrieval**: Fetch real-time weather data from Firestore for a specified location.
- **User Information Collection**: Collect and store user information (name, address, phone number) in Firestore.
- **Chatbot Responses**: Respond to user queries using a trained neural network model.
- **Time and Date Information**: Provide current time and date information upon request.
- **List App Users**: Retrieve and display a list of app users stored in Firestore.

## Requirements

- Python 3.7+
- Flask
- Firebase Admin SDK
- PyTorch
- NumPy

## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/nisanray/AI_ChatBot_Model
    cd AI_ChatBot_Model
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up Firebase**:
    - Download your Firebase service account key and save it as `serviceAccountKey.json` in the project root directory.
    - Ensure your Firestore database is set up with the required collections (`weather_updates`, `app_users`).

5. **Prepare model and data files**:
    - Ensure `model.pth`, `intents.json`, and `train_data.json` are in the project root directory.

## Running the Application

1. **Start the Flask server**:
    ```sh
    python app.py
    ```

2. **Access the application**:
    - The application will be running at `http://127.0.0.1:5000/`.

## API Endpoints

- **Chatbot Response**: `/chatbot` (POST)
    - Request:
        ```json
        {
            "message": "your message here",
            "user_id": "optional user id"
        }
        ```
    - Response:
        ```json
        {
            "response": "chatbot response here"
        }
        ```

- **List Users**: `/users` (GET)
    - Response:
        ```json
        {
            "users": [
                {
                    "user_id": "user id",
                    "name": "user name",
                    "address": "user address",
                    "phone": "user phone",
                    "timestamp": "timestamp"
                },
               "..."
            ]
        }
        ```

## License

This project is licensed under the Nisan Ray.
