# run.py
from routes import create_app

app = create_app()
# HOST_IP = ""
HOST_PORT = 5000

if __name__ == '__main__':
    app.run(debug=True, port=HOST_PORT)
