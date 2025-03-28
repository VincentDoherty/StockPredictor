import unittest
from app import app, get_db_connection
from werkzeug.security import generate_password_hash

class FlaskTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Set up the database and create a test user
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, username VARCHAR(255) UNIQUE NOT NULL, password_hash TEXT NOT NULL)")
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING", ('testuser', generate_password_hash('testpassword')))
        conn.commit()
        cursor.close()
        conn.close()

    def tearDown(self):
        # Clean up the specific user
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = %s", ('testuser',))
        cursor.execute("DELETE FROM users WHERE username = %s", ('newuser',))
        conn.commit()
        cursor.close()
        conn.close()

    def test_registration(self):
        # Test registration with a new username
        response = self.app.post('/register', data=dict(username='newuser', password='newpassword'), follow_redirects=True)
        self.assertIn(b'Registration successful', response.data)  # Check for success message

        # Test registration with an existing username
        response = self.app.post('/register', data=dict(username='testuser', password='testpassword'), follow_redirects=True)
        self.assertIn(b'Username already exists', response.data)  # Check for error message

if __name__ == '__main__':
    unittest.main()